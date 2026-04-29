// Copyright 2026 The Cloud Hypervisor Authors.
//
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::io::{self, Result};
use std::marker::PhantomData;
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use io_uring::{cqueue, opcode, squeue, types, IoUring};
use vmm_sys_util::epoll::EventSet;
use vmm_sys_util::event::{
    new_event_consumer_and_notifier, EventConsumer, EventFlag, EventNotifier,
};

use super::backend::VhostUserBackend;
use super::vring::VringT;

const USER_DATA_KIND_SHIFT: u64 = 60;
const USER_DATA_PAYLOAD_MASK: u64 = (1u64 << USER_DATA_KIND_SHIFT) - 1;

const USER_DATA_KIND_BACKEND: u64 = 0;
const USER_DATA_KIND_POLL: u64 = 1;
const USER_DATA_KIND_CANCEL: u64 = 2;

const COMMAND_POLL_TOKEN: u64 = 0;
const EXIT_POLL_TOKEN: u64 = 1;
const FIRST_BACKEND_POLL_TOKEN: u64 = 2;

/// Mask available to backend implementations for io_uring SQE `user_data`.
pub const VHOST_USER_BACKEND_USER_DATA_MASK: u64 = USER_DATA_PAYLOAD_MASK;

/// Configuration for an io_uring based vring worker.
#[derive(Clone, Debug)]
pub struct IoUringConfig {
    /// Number of submission queue entries.
    pub entries: u32,
    /// Optional completion queue size. This should usually be larger than `entries`.
    pub cq_entries: Option<u32>,
    /// Try multishot poll first for eventfds/listener fds.
    pub multishot_poll: bool,
    /// Enable SQPOLL with the specified idle timeout in milliseconds.
    pub sqpoll_idle_ms: Option<u32>,
}

impl Default for IoUringConfig {
    fn default() -> Self {
        IoUringConfig {
            entries: 256,
            cq_entries: Some(512),
            multishot_poll: true,
            sqpoll_idle_ms: None,
        }
    }
}

impl IoUringConfig {
    fn build_ring(&self) -> io::Result<IoUring> {
        let mut builder = IoUring::builder();

        if let Some(cq_entries) = self.cq_entries {
            builder.setup_cqsize(cq_entries);
        }

        if let Some(idle) = self.sqpoll_idle_ms {
            builder.setup_sqpoll(idle);
        }

        builder.build(self.entries)
    }
}

/// Completion for an SQE submitted by a backend through [`VhostUserBackendContext`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IoUringCompletion {
    /// Backend-provided user data with the library-reserved high bits stripped.
    pub user_data: u64,
    /// Operation result from the CQE.
    pub result: i32,
    /// CQE flags.
    pub flags: u32,
}

/// Event delivered to an io_uring-aware backend callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VhostUserBackendEvent {
    /// A queue kick or backend-registered file descriptor became ready.
    Fd { device_event: u16, evset: EventSet },
    /// An SQE submitted by the backend completed.
    IoUringCompletion(IoUringCompletion),
}

/// Context passed to backend callbacks running on an io_uring worker thread.
pub struct VhostUserBackendContext<'a> {
    ring: &'a mut IoUring,
    thread_id: usize,
}

impl<'a> VhostUserBackendContext<'a> {
    fn new(ring: &'a mut IoUring, thread_id: usize) -> Self {
        VhostUserBackendContext { ring, thread_id }
    }

    /// Worker thread id associated with this io_uring.
    pub fn thread_id(&self) -> usize {
        self.thread_id
    }

    /// Submit one backend SQE using the shared worker io_uring.
    ///
    /// The top 4 bits of `user_data` are reserved by the library and must be zero.
    pub fn submit_entry(&mut self, entry: squeue::Entry, user_data: u64) -> io::Result<()> {
        let user_data = encode_user_data(USER_DATA_KIND_BACKEND, user_data)?;
        push_entry(self.ring, entry.user_data(user_data))
    }

    /// Submit several backend SQEs using the shared worker io_uring.
    ///
    /// The top 4 bits of each `user_data` value are reserved by the library and must be zero.
    pub fn submit_entries<I>(&mut self, entries: I) -> io::Result<()>
    where
        I: IntoIterator<Item = (squeue::Entry, u64)>,
    {
        for (entry, user_data) in entries {
            self.submit_entry(entry, user_data)?;
        }
        Ok(())
    }

    /// Submit all currently queued SQEs to the kernel.
    pub fn submit(&mut self) -> io::Result<usize> {
        self.ring.submit()
    }

    /// Register fixed files for this ring.
    pub fn register_files(&mut self, fds: &[RawFd]) -> io::Result<()> {
        self.ring.submitter().register_files(fds)
    }

    /// Register a sparse fixed file table for this ring.
    pub fn register_files_sparse(&mut self, nr: u32) -> io::Result<()> {
        self.ring.submitter().register_files_sparse(nr)
    }

    /// Update fixed files for this ring.
    pub fn register_files_update(&mut self, offset: u32, fds: &[RawFd]) -> io::Result<usize> {
        self.ring.submitter().register_files_update(offset, fds)
    }

    /// Unregister fixed files for this ring.
    pub fn unregister_files(&mut self) -> io::Result<()> {
        self.ring.submitter().unregister_files()
    }

    /// Register fixed buffers for this ring.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all iovec ranges stay valid until they are unregistered or the
    /// ring is destroyed.
    pub unsafe fn register_buffers(&mut self, bufs: &[libc::iovec]) -> io::Result<()> {
        unsafe { self.ring.submitter().register_buffers(bufs) }
    }

    /// Register a sparse fixed buffer table for this ring.
    pub fn register_buffers_sparse(&mut self, nr: u32) -> io::Result<()> {
        self.ring.submitter().register_buffers_sparse(nr)
    }

    /// Unregister fixed buffers for this ring.
    pub fn unregister_buffers(&mut self) -> io::Result<()> {
        self.ring.submitter().unregister_buffers()
    }
}

/// Errors related to vring io_uring event handling.
#[derive(Debug)]
pub enum VringIoUringError {
    /// Failed to create an io_uring instance.
    IoUringCreate(io::Error),
    /// Failed to create an internal eventfd.
    CreateEvent(io::Error),
    /// Failed to queue an SQE.
    QueueSubmission(io::Error),
    /// Failed while submitting or waiting for events.
    SubmitAndWait(io::Error),
    /// Failed to read an internal event.
    ConsumeEvent(io::Error),
    /// Failed to read the event from kick EventFd.
    HandleEventReadKick(io::Error),
    /// Failed to handle the event from the backend.
    HandleEventBackendHandling(io::Error),
    /// The io_uring worker was already started.
    AlreadyStarted,
}

impl Display for VringIoUringError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            VringIoUringError::IoUringCreate(e) => write!(f, "cannot create io_uring: {e}"),
            VringIoUringError::CreateEvent(e) => write!(f, "cannot create eventfd: {e}"),
            VringIoUringError::QueueSubmission(e) => write!(f, "cannot queue SQE: {e}"),
            VringIoUringError::SubmitAndWait(e) => {
                write!(f, "failed to submit or wait for io_uring event: {e}")
            }
            VringIoUringError::ConsumeEvent(e) => write!(f, "cannot consume eventfd: {e}"),
            VringIoUringError::HandleEventReadKick(e) => {
                write!(f, "cannot read vring kick event: {e}")
            }
            VringIoUringError::HandleEventBackendHandling(e) => {
                write!(f, "failed to handle io_uring event: {e}")
            }
            VringIoUringError::AlreadyStarted => write!(f, "io_uring worker already started"),
        }
    }
}

impl std::error::Error for VringIoUringError {}

/// Result of vring io_uring operations.
pub type VringIoUringResult<T> = std::result::Result<T, VringIoUringError>;

#[derive(Clone, Copy, Debug)]
struct RegisteredEvent {
    token: u64,
    evset: EventSet,
    data: u64,
}

#[derive(Clone, Copy, Debug)]
enum PollKind {
    Command,
    Exit,
    BackendFd,
}

#[derive(Clone, Copy, Debug)]
struct PollRegistration {
    fd: RawFd,
    evset: EventSet,
    data: u64,
    kind: PollKind,
    multishot: bool,
}

#[derive(Clone, Copy, Debug)]
enum IoUringCommand {
    Add {
        token: u64,
        fd: RawFd,
        evset: EventSet,
        data: u64,
    },
    Remove {
        token: u64,
    },
}

/// io_uring event handler to manage and process registered file descriptors.
pub struct VringIoUringHandler<T: VhostUserBackend> {
    ring: Mutex<Option<IoUring>>,
    backend: T,
    vrings: Vec<T::Vring>,
    thread_id: usize,
    config: IoUringConfig,
    exit_event_fd: Option<EventNotifier>,
    exit_event_consumer: Mutex<Option<EventConsumer>>,
    command_consumer: Mutex<Option<EventConsumer>>,
    command_notifier: EventNotifier,
    commands: Mutex<VecDeque<IoUringCommand>>,
    registrations: Mutex<HashMap<RawFd, RegisteredEvent>>,
    next_token: AtomicU64,
    phantom: PhantomData<T::Bitmap>,
}

impl<T: VhostUserBackend> VringIoUringHandler<T> {
    /// Send `exit event` to break the event loop.
    pub fn send_exit_event(&self) {
        if let Some(eventfd) = self.exit_event_fd.as_ref() {
            let _ = eventfd.notify();
        }
    }
}

impl<T> VringIoUringHandler<T>
where
    T: VhostUserBackend,
{
    /// Create a `VringIoUringHandler` instance.
    pub(crate) fn new(
        backend: T,
        vrings: Vec<T::Vring>,
        thread_id: usize,
        config: IoUringConfig,
    ) -> VringIoUringResult<Self> {
        let ring = config
            .build_ring()
            .map_err(VringIoUringError::IoUringCreate)?;
        let exit_event_fd = backend.exit_event(thread_id);
        let (exit_event_consumer, exit_event_fd) = if let Some((consumer, notifier)) = exit_event_fd
        {
            (Some(consumer), Some(notifier))
        } else {
            (None, None)
        };
        let (command_consumer, command_notifier) =
            new_event_consumer_and_notifier(EventFlag::NONBLOCK | EventFlag::CLOEXEC)
                .map_err(VringIoUringError::CreateEvent)?;

        Ok(VringIoUringHandler {
            ring: Mutex::new(Some(ring)),
            backend,
            vrings,
            thread_id,
            config,
            exit_event_fd,
            exit_event_consumer: Mutex::new(exit_event_consumer),
            command_consumer: Mutex::new(Some(command_consumer)),
            command_notifier,
            commands: Mutex::new(VecDeque::new()),
            registrations: Mutex::new(HashMap::new()),
            next_token: AtomicU64::new(FIRST_BACKEND_POLL_TOKEN),
            phantom: PhantomData,
        })
    }

    /// Register an event into the io_uring worker.
    ///
    /// When this event is later triggered, the backend implementation of
    /// `handle_event_with_context` will be called.
    pub fn register_listener(&self, fd: RawFd, ev_type: EventSet, data: u64) -> Result<()> {
        if data <= self.backend.num_queues() as u64 {
            Err(io::Error::from_raw_os_error(libc::EINVAL))
        } else {
            self.register_event(fd, ev_type, data)
        }
    }

    /// Unregister an event from the io_uring worker.
    pub fn unregister_listener(&self, fd: RawFd, ev_type: EventSet, data: u64) -> Result<()> {
        if data <= self.backend.num_queues() as u64 {
            Err(io::Error::from_raw_os_error(libc::EINVAL))
        } else {
            self.unregister_event(fd, ev_type, data)
        }
    }

    pub(crate) fn register_event(&self, fd: RawFd, ev_type: EventSet, data: u64) -> Result<()> {
        if fd < 0 {
            return Err(io::Error::from_raw_os_error(libc::EBADF));
        }

        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        let mut registrations = self.registrations.lock().unwrap();
        if registrations.contains_key(&fd) {
            return Err(io::Error::from_raw_os_error(libc::EEXIST));
        }
        registrations.insert(
            fd,
            RegisteredEvent {
                token,
                evset: ev_type,
                data,
            },
        );
        drop(registrations);

        self.enqueue_command(IoUringCommand::Add {
            token,
            fd,
            evset: ev_type,
            data,
        })
    }

    pub(crate) fn unregister_event(&self, fd: RawFd, ev_type: EventSet, data: u64) -> Result<()> {
        let mut registrations = self.registrations.lock().unwrap();
        let Some(registered) = registrations.get(&fd).copied() else {
            return Err(io::Error::from_raw_os_error(libc::ENOENT));
        };
        if registered.data != data || registered.evset != ev_type {
            return Err(io::Error::from_raw_os_error(libc::ENOENT));
        }
        registrations.remove(&fd);
        drop(registrations);

        self.enqueue_command(IoUringCommand::Remove {
            token: registered.token,
        })
    }

    fn enqueue_command(&self, command: IoUringCommand) -> Result<()> {
        self.commands.lock().unwrap().push_back(command);
        self.command_notifier.notify()
    }

    /// Run the io_uring loop to handle all pending events on registered fds and backend SQEs.
    ///
    /// The event loop will be terminated once an event is received from the `exit event fd`
    /// associated with the backend.
    pub(crate) fn run(&self) -> VringIoUringResult<()> {
        let ring = self
            .ring
            .lock()
            .unwrap()
            .take()
            .ok_or(VringIoUringError::AlreadyStarted)?;
        let command_consumer = self
            .command_consumer
            .lock()
            .unwrap()
            .take()
            .ok_or(VringIoUringError::AlreadyStarted)?;
        let exit_event_consumer = self.exit_event_consumer.lock().unwrap().take();

        let mut worker = VringIoUringWorker {
            handler: self,
            ring,
            command_consumer,
            exit_event_consumer,
            active_polls: HashMap::new(),
        };

        worker.run()
    }
}

struct VringIoUringWorker<'a, T: VhostUserBackend> {
    handler: &'a VringIoUringHandler<T>,
    ring: IoUring,
    command_consumer: EventConsumer,
    exit_event_consumer: Option<EventConsumer>,
    active_polls: HashMap<u64, PollRegistration>,
}

impl<T: VhostUserBackend> VringIoUringWorker<'_, T> {
    fn run(&mut self) -> VringIoUringResult<()> {
        self.add_internal_poll(
            COMMAND_POLL_TOKEN,
            self.command_consumer.as_raw_fd(),
            EventSet::IN,
            PollKind::Command,
        )?;

        if let Some(consumer) = self.exit_event_consumer.as_ref() {
            self.add_internal_poll(
                EXIT_POLL_TOKEN,
                consumer.as_raw_fd(),
                EventSet::IN,
                PollKind::Exit,
            )?;
        }

        loop {
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(VringIoUringError::SubmitAndWait(e)),
            }

            let completions = self.collect_completions();
            for (user_data, result, flags) in completions {
                if self.dispatch_completion(user_data, result, flags)? {
                    return Ok(());
                }
            }
        }
    }

    fn collect_completions(&mut self) -> Vec<(u64, i32, u32)> {
        let mut completions = Vec::new();
        let completion_queue = self.ring.completion();
        for cqe in completion_queue {
            completions.push((cqe.user_data(), cqe.result(), cqe.flags()));
        }
        completions
    }

    fn dispatch_completion(
        &mut self,
        user_data: u64,
        result: i32,
        flags: u32,
    ) -> VringIoUringResult<bool> {
        match decode_user_data_kind(user_data) {
            USER_DATA_KIND_BACKEND => {
                let completion = IoUringCompletion {
                    user_data: user_data & USER_DATA_PAYLOAD_MASK,
                    result,
                    flags,
                };
                self.dispatch_backend_event(VhostUserBackendEvent::IoUringCompletion(completion))?;
                Ok(false)
            }
            USER_DATA_KIND_POLL => {
                self.dispatch_poll_completion(user_data & USER_DATA_PAYLOAD_MASK, result, flags)
            }
            USER_DATA_KIND_CANCEL => Ok(false),
            _ => Ok(false),
        }
    }

    fn dispatch_poll_completion(
        &mut self,
        token: u64,
        result: i32,
        flags: u32,
    ) -> VringIoUringResult<bool> {
        let Some(mut registration) = self.active_polls.get(&token).copied() else {
            return Ok(false);
        };

        if result == -libc::EINVAL && registration.multishot {
            registration.multishot = false;
            self.active_polls.insert(token, registration);
            self.arm_poll(token, registration)?;
            return Ok(false);
        }

        if result < 0 {
            return Err(VringIoUringError::SubmitAndWait(
                io::Error::from_raw_os_error(-result),
            ));
        }

        let evset = poll_flags_to_event_set(result as u32);
        let more_events = registration.multishot && cqueue::more(flags);

        let exit = match registration.kind {
            PollKind::Command => {
                drain_event(&self.command_consumer)?;
                self.drain_commands()?;
                false
            }
            PollKind::Exit => {
                if let Some(consumer) = self.exit_event_consumer.as_ref() {
                    consume_event(consumer)?;
                }
                true
            }
            PollKind::BackendFd => {
                self.dispatch_fd_event(registration.data as u16, evset)?;
                false
            }
        };

        if !exit && !more_events && self.active_polls.contains_key(&token) {
            self.arm_poll(token, registration)?;
        }

        Ok(exit)
    }

    fn dispatch_fd_event(&mut self, device_event: u16, evset: EventSet) -> VringIoUringResult<()> {
        if (device_event as usize) < self.handler.vrings.len() {
            let vring = &self.handler.vrings[device_event as usize];
            let enabled = vring
                .read_kick()
                .map_err(VringIoUringError::HandleEventReadKick)?;

            if !enabled {
                return Ok(());
            }
        }

        self.dispatch_backend_event(VhostUserBackendEvent::Fd {
            device_event,
            evset,
        })
    }

    fn dispatch_backend_event(&mut self, event: VhostUserBackendEvent) -> VringIoUringResult<()> {
        let mut context = VhostUserBackendContext::new(&mut self.ring, self.handler.thread_id);

        self.handler
            .backend
            .handle_event_with_context(
                event,
                &self.handler.vrings,
                self.handler.thread_id,
                &mut context,
            )
            .map_err(VringIoUringError::HandleEventBackendHandling)
    }

    fn drain_commands(&mut self) -> VringIoUringResult<()> {
        loop {
            let command = self.handler.commands.lock().unwrap().pop_front();
            let Some(command) = command else {
                break;
            };

            match command {
                IoUringCommand::Add {
                    token,
                    fd,
                    evset,
                    data,
                } => {
                    let registration = PollRegistration {
                        fd,
                        evset,
                        data,
                        kind: PollKind::BackendFd,
                        multishot: self.handler.config.multishot_poll,
                    };
                    self.active_polls.insert(token, registration);
                    self.arm_poll(token, registration)?;
                }
                IoUringCommand::Remove { token } => {
                    if self.active_polls.remove(&token).is_some() {
                        self.cancel_poll(token)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn add_internal_poll(
        &mut self,
        token: u64,
        fd: RawFd,
        evset: EventSet,
        kind: PollKind,
    ) -> VringIoUringResult<()> {
        let registration = PollRegistration {
            fd,
            evset,
            data: token,
            kind,
            multishot: self.handler.config.multishot_poll,
        };

        self.active_polls.insert(token, registration);
        self.arm_poll(token, registration)
    }

    fn arm_poll(&mut self, token: u64, registration: PollRegistration) -> VringIoUringResult<()> {
        let entry = opcode::PollAdd::new(
            types::Fd(registration.fd),
            event_set_to_poll_flags(registration.evset),
        )
        .multi(registration.multishot)
        .build()
        .user_data(
            encode_user_data(USER_DATA_KIND_POLL, token)
                .map_err(VringIoUringError::QueueSubmission)?,
        );

        push_entry(&mut self.ring, entry).map_err(VringIoUringError::QueueSubmission)
    }

    fn cancel_poll(&mut self, token: u64) -> VringIoUringResult<()> {
        let entry = opcode::PollRemove::new(
            encode_user_data(USER_DATA_KIND_POLL, token)
                .map_err(VringIoUringError::QueueSubmission)?,
        )
        .build()
        .user_data(
            encode_user_data(USER_DATA_KIND_CANCEL, token)
                .map_err(VringIoUringError::QueueSubmission)?,
        );

        push_entry(&mut self.ring, entry).map_err(VringIoUringError::QueueSubmission)
    }
}

fn push_entry(ring: &mut IoUring, entry: squeue::Entry) -> io::Result<()> {
    // SAFETY: Callers provide SQEs whose referenced fds and buffers must remain valid until
    // completion. The io_uring crate requires `unsafe` here because it cannot verify that.
    let result = {
        let mut submission = ring.submission();
        // SAFETY: See comment above.
        unsafe { submission.push(&entry) }
    };

    match result {
        Ok(()) => Ok(()),
        Err(_) => {
            ring.submit()?;
            // SAFETY: Same as above.
            let result = {
                let mut submission = ring.submission();
                unsafe { submission.push(&entry) }
            };
            result
                .map_err(|_| io::Error::new(io::ErrorKind::WouldBlock, "submission queue is full"))
        }
    }
}

fn encode_user_data(kind: u64, payload: u64) -> io::Result<u64> {
    if kind > 0xf || payload > USER_DATA_PAYLOAD_MASK {
        Err(io::Error::from_raw_os_error(libc::EINVAL))
    } else {
        Ok((kind << USER_DATA_KIND_SHIFT) | payload)
    }
}

fn decode_user_data_kind(user_data: u64) -> u64 {
    user_data >> USER_DATA_KIND_SHIFT
}

fn consume_event(consumer: &EventConsumer) -> VringIoUringResult<()> {
    consumer.consume().map_err(VringIoUringError::ConsumeEvent)
}

fn drain_event(consumer: &EventConsumer) -> VringIoUringResult<()> {
    loop {
        match consumer.consume() {
            Ok(()) => continue,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(e) => return Err(VringIoUringError::ConsumeEvent(e)),
        }
    }
}

fn event_set_to_poll_flags(evset: EventSet) -> u32 {
    let mut flags = 0;
    if evset.contains(EventSet::IN) {
        flags |= libc::POLLIN as u32;
    }
    if evset.contains(EventSet::OUT) {
        flags |= libc::POLLOUT as u32;
    }
    if evset.contains(EventSet::ERROR) {
        flags |= libc::POLLERR as u32;
    }
    if evset.contains(EventSet::HANG_UP) {
        flags |= libc::POLLHUP as u32;
    }
    if evset.contains(EventSet::PRIORITY) {
        flags |= libc::POLLPRI as u32;
    }
    if evset.contains(EventSet::READ_HANG_UP) {
        flags |= libc::POLLRDHUP as u32;
    }
    flags
}

fn poll_flags_to_event_set(flags: u32) -> EventSet {
    let mut evset = EventSet::empty();
    if flags & libc::POLLIN as u32 != 0 {
        evset |= EventSet::IN;
    }
    if flags & libc::POLLOUT as u32 != 0 {
        evset |= EventSet::OUT;
    }
    if flags & libc::POLLERR as u32 != 0 {
        evset |= EventSet::ERROR;
    }
    if flags & libc::POLLHUP as u32 != 0 {
        evset |= EventSet::HANG_UP;
    }
    if flags & libc::POLLPRI as u32 != 0 {
        evset |= EventSet::PRIORITY;
    }
    if flags & libc::POLLRDHUP as u32 != 0 {
        evset |= EventSet::READ_HANG_UP;
    }
    evset
}

#[cfg(test)]
mod tests {
    use super::super::backend::tests::MockVhostBackend;
    use super::super::vring::VringRwLock;
    use super::*;
    use std::fs::File;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};
    use vhost::vhost_user::message::{
        VhostTransferStateDirection, VhostTransferStatePhase, VhostUserProtocolFeatures,
        VhostUserShMemConfig, VhostUserSharedMsg,
    };
    use vhost::vhost_user::{Backend, GpuBackend};
    use vm_memory::{GuestAddress, GuestMemoryAtomic, GuestMemoryMmap};
    use vmm_sys_util::event::{new_event_consumer_and_notifier, EventFlag};

    fn io_uring_available() -> bool {
        IoUring::new(2).is_ok()
    }

    #[test]
    fn test_user_data_encoding() {
        assert_eq!(
            encode_user_data(USER_DATA_KIND_BACKEND, 0x42).unwrap(),
            0x42
        );
        assert!(encode_user_data(USER_DATA_KIND_BACKEND, USER_DATA_PAYLOAD_MASK + 1).is_err());
    }

    #[test]
    fn test_vring_iouring_handler_registration() {
        if !io_uring_available() {
            return;
        }

        let mem = GuestMemoryAtomic::new(
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0x100000), 0x10000)]).unwrap(),
        );
        let vring = VringRwLock::new(mem, 0x1000).unwrap();
        let backend = Arc::new(Mutex::new(MockVhostBackend::new()));

        let handler =
            VringIoUringHandler::new(backend, vec![vring], 0x1, IoUringConfig::default()).unwrap();

        let (consumer, _notifier) = new_event_consumer_and_notifier(EventFlag::empty()).unwrap();
        handler
            .register_listener(consumer.as_raw_fd(), EventSet::IN, 3)
            .unwrap();
        handler
            .register_listener(consumer.as_raw_fd(), EventSet::IN, 3)
            .unwrap_err();
        handler
            .register_listener(consumer.as_raw_fd(), EventSet::IN, 1)
            .unwrap_err();
        handler
            .unregister_listener(consumer.as_raw_fd(), EventSet::IN, 3)
            .unwrap();
        handler
            .unregister_listener(consumer.as_raw_fd(), EventSet::IN, 3)
            .unwrap_err();
        handler
            .unregister_listener(consumer.as_raw_fd(), EventSet::IN, 1)
            .unwrap_err();
    }

    struct CompletionBackend {
        fd_consumer: EventConsumer,
        fd_events: AtomicU64,
        completions: AtomicU64,
        exit_consumer: EventConsumer,
        exit_notifier: EventNotifier,
    }

    impl CompletionBackend {
        fn new(fd_consumer: EventConsumer) -> Self {
            let (exit_consumer, exit_notifier) =
                new_event_consumer_and_notifier(EventFlag::NONBLOCK).unwrap();

            CompletionBackend {
                fd_consumer,
                fd_events: AtomicU64::new(0),
                completions: AtomicU64::new(0),
                exit_consumer,
                exit_notifier,
            }
        }
    }

    impl VhostUserBackend for CompletionBackend {
        type Bitmap = ();
        type Vring = VringRwLock;

        fn num_queues(&self) -> usize {
            1
        }

        fn max_queue_size(&self) -> usize {
            256
        }

        fn features(&self) -> u64 {
            0xffff_ffff_ffff_ffff
        }

        fn protocol_features(&self) -> VhostUserProtocolFeatures {
            VhostUserProtocolFeatures::all()
        }

        fn set_event_idx(&self, _enabled: bool) {}

        fn update_memory(&self, _mem: super::super::GM<Self::Bitmap>) -> Result<()> {
            Ok(())
        }

        fn queues_per_thread(&self) -> Vec<u64> {
            vec![1]
        }

        fn exit_event(&self, _thread_index: usize) -> Option<(EventConsumer, EventNotifier)> {
            Some((
                self.exit_consumer.try_clone().unwrap(),
                self.exit_notifier.try_clone().unwrap(),
            ))
        }

        fn handle_event(
            &self,
            _device_event: u16,
            _evset: EventSet,
            _vrings: &[Self::Vring],
            _thread_id: usize,
        ) -> Result<()> {
            Ok(())
        }

        fn handle_event_with_context(
            &self,
            event: VhostUserBackendEvent,
            _vrings: &[Self::Vring],
            _thread_id: usize,
            context: &mut VhostUserBackendContext<'_>,
        ) -> Result<()> {
            match event {
                VhostUserBackendEvent::Fd { .. } => {
                    self.fd_consumer.consume()?;
                    self.fd_events.fetch_add(1, Ordering::Relaxed);
                    context.submit_entry(opcode::Nop::new().build(), 0x123)?;
                    context.submit()?;
                }
                VhostUserBackendEvent::IoUringCompletion(completion) => {
                    if completion.user_data == 0x123 && completion.result == 0 {
                        self.completions.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Ok(())
        }

        fn set_backend_req_fd(&self, _backend: Backend) {}

        fn get_shared_object(&self, _uuid: VhostUserSharedMsg) -> Result<File> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "unsupported"))
        }

        fn set_gpu_socket(&self, _gpu_backend: GpuBackend) -> Result<()> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "unsupported"))
        }

        fn set_device_state_fd(
            &self,
            _direction: VhostTransferStateDirection,
            _phase: VhostTransferStatePhase,
            _file: File,
        ) -> Result<Option<File>> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "unsupported"))
        }

        fn check_device_state(&self) -> Result<()> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "unsupported"))
        }

        fn get_shmem_config(&self) -> Result<VhostUserShMemConfig> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "unsupported"))
        }
    }

    #[test]
    fn test_backend_iouring_completion_dispatch() {
        if !io_uring_available() {
            return;
        }

        let mem = GuestMemoryAtomic::new(
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0x100000), 0x10000)]).unwrap(),
        );
        let vring = VringRwLock::new(mem, 0x1000).unwrap();
        let (consumer, notifier) = new_event_consumer_and_notifier(EventFlag::NONBLOCK).unwrap();
        let backend = Arc::new(CompletionBackend::new(consumer.try_clone().unwrap()));
        let handler = Arc::new(
            VringIoUringHandler::new(backend.clone(), vec![vring], 0, IoUringConfig::default())
                .unwrap(),
        );

        let runner = {
            let handler = handler.clone();
            thread::spawn(move || handler.run())
        };

        handler
            .register_listener(consumer.as_raw_fd(), EventSet::IN, 2)
            .unwrap();
        notifier.notify().unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if backend.completions.load(Ordering::Relaxed) == 1 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        handler.send_exit_event();
        runner.join().unwrap().unwrap();

        assert_eq!(backend.fd_events.load(Ordering::Relaxed), 1);
        assert_eq!(backend.completions.load(Ordering::Relaxed), 1);
    }
}
