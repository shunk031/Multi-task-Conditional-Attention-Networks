# Multi-task Conditional Attention Networks

A prototype version of our submitted paper: `Conversion Prediction Using Multi-task Conditional Attention Networks to Support the Creation of Effective Ad Creatives`.

## Setup using Docker

```shell
$ docker build -t multi-task-cond-net-env .
$ docker create -it -v /data:/data --name datavolume busybox
$ docker run -it -p 8888:8888 --runtime=nvidia --volumes-from datavolume --rm --name multi-task-cond-net multi-task-cond-net-env
```
