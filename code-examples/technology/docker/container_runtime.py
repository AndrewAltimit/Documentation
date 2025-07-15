"""
Container Runtime Architecture Implementation

Low-level container runtime following OCI (Open Container Initiative) specification,
including bundle creation, namespace configuration, and resource management.
"""

import asyncio
import json
import os
import subprocess
from typing import Dict, List, Optional, Any
import ipaddress


class ContainerRuntime:
    """Low-level container runtime implementation following OCI spec"""
    
    def __init__(self, runtime_path: str = "/usr/bin/runc"):
        self.runtime_path = runtime_path
        self.oci_version = "1.0.2"
    
    def create_oci_bundle(self, rootfs_path: str, config: Dict[str, Any]) -> str:
        """
        Create OCI bundle with config.json and rootfs
        
        OCI Runtime Specification:
        - config.json: Container configuration
        - rootfs/: Root filesystem
        """
        bundle_dir = f"/run/containers/{config['hostname']}"
        os.makedirs(f"{bundle_dir}/rootfs", exist_ok=True)
        
        # OCI config.json structure
        oci_config = {
            "ociVersion": self.oci_version,
            "process": {
                "terminal": config.get("tty", False),
                "user": {
                    "uid": config.get("uid", 0),
                    "gid": config.get("gid", 0)
                },
                "args": config.get("cmd", ["/bin/sh"]),
                "env": [
                    f"{k}={v}" for k, v in config.get("env", {}).items()
                ],
                "cwd": config.get("working_dir", "/"),
                "capabilities": {
                    "bounding": self._get_capabilities(config),
                    "effective": self._get_capabilities(config),
                    "inheritable": self._get_capabilities(config),
                    "permitted": self._get_capabilities(config),
                    "ambient": self._get_capabilities(config)
                },
                "rlimits": [
                    {
                        "type": "RLIMIT_NOFILE",
                        "hard": 1024,
                        "soft": 1024
                    }
                ],
                "noNewPrivileges": True
            },
            "root": {
                "path": "rootfs",
                "readonly": config.get("readonly_rootfs", False)
            },
            "hostname": config.get("hostname", "container"),
            "mounts": self._get_mounts(config),
            "linux": {
                "resources": self._get_resources(config),
                "namespaces": self._get_namespaces(),
                "seccomp": self._get_seccomp_config(),
                "maskedPaths": [
                    "/proc/kcore",
                    "/proc/latency_stats",
                    "/proc/timer_list",
                    "/proc/timer_stats",
                    "/proc/sched_debug",
                    "/sys/firmware"
                ],
                "readonlyPaths": [
                    "/proc/asound",
                    "/proc/bus",
                    "/proc/fs",
                    "/proc/irq",
                    "/proc/sys",
                    "/proc/sysrq-trigger"
                ]
            }
        }
        
        # Write config.json
        with open(f"{bundle_dir}/config.json", 'w') as f:
            json.dump(oci_config, f, indent=2)
        
        # Copy rootfs
        subprocess.run(["cp", "-a", f"{rootfs_path}/.", f"{bundle_dir}/rootfs/"])
        
        return bundle_dir
    
    def _get_namespaces(self) -> List[Dict[str, str]]:
        """Get namespace configuration"""
        return [
            {"type": "pid"},
            {"type": "network"},
            {"type": "ipc"},
            {"type": "uts"},
            {"type": "mount"},
            {"type": "user"},
            {"type": "cgroup"}
        ]
    
    def _get_resources(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get resource limits configuration"""
        return {
            "devices": "allow",
            "memory": {
                "limit": config.get("memory_limit", 536870912),  # 512MB default
                "reservation": config.get("memory_reservation", 268435456),
                "swap": config.get("memory_swap", 536870912),
                "kernel": config.get("kernel_memory", -1),
                "kernelTCP": config.get("kernel_memory_tcp", -1),
                "swappiness": config.get("memory_swappiness", 60),
                "disableOOMKiller": config.get("oom_kill_disable", False)
            },
            "cpu": {
                "shares": config.get("cpu_shares", 1024),
                "quota": config.get("cpu_quota", -1),
                "period": config.get("cpu_period", 100000),
                "realtimeRuntime": config.get("cpu_rt_runtime", 0),
                "realtimePeriod": config.get("cpu_rt_period", 0),
                "cpus": config.get("cpuset_cpus", ""),
                "mems": config.get("cpuset_mems", "")
            },
            "pids": {
                "limit": config.get("pids_limit", 32768)
            },
            "blockIO": {
                "weight": config.get("blkio_weight", 0),
                "leafWeight": config.get("blkio_leaf_weight", 0),
                "weightDevice": [],
                "throttleReadBpsDevice": [],
                "throttleWriteBpsDevice": [],
                "throttleReadIOPSDevice": [],
                "throttleWriteIOPSDevice": []
            },
            "hugepageLimits": [],
            "network": {
                "classID": config.get("net_cls_classid", 0),
                "priorities": []
            }
        }
    
    def _get_capabilities(self, config: Dict[str, Any]) -> List[str]:
        """Get Linux capabilities for container"""
        default_caps = [
            "CAP_CHOWN",
            "CAP_DAC_OVERRIDE",
            "CAP_FOWNER",
            "CAP_FSETID",
            "CAP_KILL",
            "CAP_SETGID",
            "CAP_SETUID",
            "CAP_SETPCAP",
            "CAP_NET_BIND_SERVICE",
            "CAP_NET_RAW",
            "CAP_SYS_CHROOT",
            "CAP_MKNOD",
            "CAP_AUDIT_WRITE",
            "CAP_SETFCAP"
        ]
        
        return config.get("capabilities", default_caps)
    
    def _get_mounts(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get mount configuration"""
        mounts = [
            {
                "destination": "/proc",
                "type": "proc",
                "source": "proc",
                "options": ["nosuid", "noexec", "nodev"]
            },
            {
                "destination": "/dev",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "strictatime", "mode=755", "size=65536k"]
            },
            {
                "destination": "/dev/pts",
                "type": "devpts",
                "source": "devpts",
                "options": ["nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620"]
            },
            {
                "destination": "/dev/shm",
                "type": "tmpfs",
                "source": "shm",
                "options": ["nosuid", "noexec", "nodev", "mode=1777", "size=65536k"]
            },
            {
                "destination": "/dev/mqueue",
                "type": "mqueue",
                "source": "mqueue",
                "options": ["nosuid", "noexec", "nodev"]
            },
            {
                "destination": "/sys",
                "type": "sysfs",
                "source": "sysfs",
                "options": ["nosuid", "noexec", "nodev", "ro"]
            }
        ]
        
        # Add custom mounts
        for mount in config.get("mounts", []):
            mounts.append(mount)
        
        return mounts
    
    def _get_seccomp_config(self) -> Dict[str, Any]:
        """Get seccomp security profile"""
        return {
            "defaultAction": "SCMP_ACT_ERRNO",
            "architectures": [
                "SCMP_ARCH_X86_64",
                "SCMP_ARCH_X86",
                "SCMP_ARCH_X32"
            ],
            "syscalls": [
                {
                    "names": [
                        "accept", "accept4", "access", "alarm", "bind", "brk",
                        "capget", "capset", "chdir", "chmod", "chown", "chown32",
                        "clock_getres", "clock_gettime", "clock_nanosleep", "close",
                        "connect", "copy_file_range", "creat", "dup", "dup2", "dup3",
                        "epoll_create", "epoll_create1", "epoll_ctl", "epoll_ctl_old",
                        "epoll_pwait", "epoll_wait", "epoll_wait_old", "eventfd",
                        "eventfd2", "execve", "execveat", "exit", "exit_group",
                        "faccessat", "fadvise64", "fadvise64_64", "fallocate",
                        "fanotify_mark", "fchdir", "fchmod", "fchmodat", "fchown",
                        "fchown32", "fchownat", "fcntl", "fcntl64", "fdatasync",
                        "fgetxattr", "flistxattr", "flock", "fork", "fremovexattr",
                        "fsetxattr", "fstat", "fstat64", "fstatat64", "fstatfs",
                        "fstatfs64", "fsync", "ftruncate", "ftruncate64", "futex",
                        "futimesat", "getcpu", "getcwd", "getdents", "getdents64",
                        "getegid", "getegid32", "geteuid", "geteuid32", "getgid",
                        "getgid32", "getgroups", "getgroups32", "getitimer", "getpeername",
                        "getpgid", "getpgrp", "getpid", "getppid", "getpriority",
                        "getrandom", "getresgid", "getresgid32", "getresuid",
                        "getresuid32", "getrlimit", "get_robust_list", "getrusage",
                        "getsid", "getsockname", "getsockopt", "get_thread_area",
                        "gettid", "gettimeofday", "getuid", "getuid32", "getxattr",
                        "inotify_add_watch", "inotify_init", "inotify_init1",
                        "inotify_rm_watch", "io_cancel", "ioctl", "io_destroy",
                        "io_getevents", "ioprio_get", "ioprio_set", "io_setup",
                        "io_submit", "kill", "lchown", "lchown32", "lgetxattr",
                        "link", "linkat", "listen", "listxattr", "llistxattr",
                        "_llseek", "lremovexattr", "lseek", "lsetxattr", "lstat",
                        "lstat64", "madvise", "memfd_create", "mincore", "mkdir",
                        "mkdirat", "mknod", "mknodat", "mlock", "mlock2", "mlockall",
                        "mmap", "mmap2", "mprotect", "mq_getsetattr", "mq_notify",
                        "mq_open", "mq_receive", "mq_send", "mq_timedreceive",
                        "mq_timedsend", "mq_unlink", "mremap", "msgctl", "msgget",
                        "msgrcv", "msgsnd", "msync", "munlock", "munlockall",
                        "munmap", "nanosleep", "newfstatat", "_newselect", "open",
                        "openat", "pause", "pipe", "pipe2", "poll", "ppoll", "prctl",
                        "pread64", "preadv", "prlimit64", "pselect6", "pwrite64",
                        "pwritev", "read", "readahead", "readlink", "readlinkat",
                        "readv", "recv", "recvfrom", "recvmmsg", "recvmsg", "remap_file_pages",
                        "removexattr", "rename", "renameat", "renameat2", "restart_syscall",
                        "rmdir", "rt_sigaction", "rt_sigpending", "rt_sigprocmask",
                        "rt_sigqueueinfo", "rt_sigreturn", "rt_sigsuspend", "rt_sigtimedwait",
                        "rt_tgsigqueueinfo", "sched_getaffinity", "sched_getattr",
                        "sched_getparam", "sched_get_priority_max", "sched_get_priority_min",
                        "sched_getscheduler", "sched_rr_get_interval", "sched_setaffinity",
                        "sched_setattr", "sched_setparam", "sched_setscheduler",
                        "sched_yield", "seccomp", "select", "semctl", "semget",
                        "semop", "semtimedop", "send", "sendfile", "sendfile64",
                        "sendmmsg", "sendmsg", "sendto", "setfsgid", "setfsgid32",
                        "setfsuid", "setfsuid32", "setgid", "setgid32", "setgroups",
                        "setgroups32", "setitimer", "setpgid", "setpriority",
                        "setregid", "setregid32", "setresgid", "setresgid32",
                        "setresuid", "setresuid32", "setreuid", "setreuid32",
                        "setrlimit", "set_robust_list", "setsid", "setsockopt",
                        "set_thread_area", "set_tid_address", "setuid", "setuid32",
                        "setxattr", "shmat", "shmctl", "shmdt", "shmget", "shutdown",
                        "sigaltstack", "signalfd", "signalfd4", "sigreturn", "socket",
                        "socketcall", "socketpair", "splice", "stat", "stat64",
                        "statfs", "statfs64", "statx", "symlink", "symlinkat",
                        "sync", "sync_file_range", "syncfs", "sysinfo", "syslog",
                        "tee", "tgkill", "time", "timer_create", "timer_delete",
                        "timerfd_create", "timerfd_gettime", "timerfd_settime",
                        "timer_getoverrun", "timer_gettime", "timer_settime", "times",
                        "tkill", "truncate", "truncate64", "ugetrlimit", "umask",
                        "uname", "unlink", "unlinkat", "utime", "utimensat", "utimes",
                        "vfork", "vmsplice", "wait4", "waitid", "waitpid", "write",
                        "writev"
                    ],
                    "action": "SCMP_ACT_ALLOW"
                }
            ]
        }
    
    async def run_container(self, bundle_path: str, container_id: str) -> int:
        """Run container using OCI runtime"""
        cmd = [
            self.runtime_path,
            "run",
            "-b", bundle_path,
            container_id
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        return process.returncode
    
    async def exec_container(self, container_id: str, cmd: List[str]) -> int:
        """Execute command in running container"""
        exec_cmd = [
            self.runtime_path,
            "exec",
            container_id
        ] + cmd
        
        process = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        return process.returncode
    
    def delete_container(self, container_id: str):
        """Delete container"""
        subprocess.run([self.runtime_path, "delete", container_id])
    
    def list_containers(self) -> List[Dict[str, Any]]:
        """List all containers"""
        result = subprocess.run(
            [self.runtime_path, "list", "--format", "json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout)
        return []


# Example usage
async def main():
    runtime = ContainerRuntime()
    
    # Container configuration
    config = {
        "hostname": "mycontainer",
        "cmd": ["/bin/sh", "-c", "echo 'Hello from container!'"],
        "env": {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "TERM": "xterm"
        },
        "working_dir": "/",
        "memory_limit": 268435456,  # 256MB
        "cpu_shares": 512,
        "pids_limit": 100
    }
    
    # Create OCI bundle
    bundle_path = runtime.create_oci_bundle("/path/to/rootfs", config)
    
    # Run container
    container_id = "mycontainer-001"
    exit_code = await runtime.run_container(bundle_path, container_id)
    print(f"Container exited with code: {exit_code}")


if __name__ == "__main__":
    asyncio.run(main())