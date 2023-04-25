## Notes
python decorator: add metadata to the process, wrap a function with our own

### Flask
get sends parameters in the header -> they are encoded but not encrypted
post receives parameters in the body -> they are encrypted

avoid handshaking for clients to be faster by keep_alive feature (resuing the channel)

process vs thread

process: atomic computation unit. They cannot share memory between them
thread: atomic unit within a process. They can share memory between them

1. unique process replicated 
2. process creates several child threads (they start and die in each request)
3. Pool of threads like a buffer, the keep_alive is treated in the process

Green threads: much more resource efficient




