We need doc on how to use the video system. A recommended workflow:

- Open one process to train headlessly
- Open another testing process (possibly on CPU) with hot-reload on to monitor the training process
  - On server: use streaming
  - On local workstation: use video; use unison to sync checkpoints with training server
