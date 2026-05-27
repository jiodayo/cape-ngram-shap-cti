#!/usr/bin/env python3
"""Generate Windows API descriptions for the desc feature pipeline.

Populates api_descriptions.json with human-readable descriptions of
Windows API functions used in malware analysis. This directly improves
the quality of desc_tfidf features used in the model.

Usage:
  python3 src/generate_api_descriptions.py \
      --api-list data/avast_ctu_cape/ngram_dataset_family_104/api.json \
      --output   data/avast_ctu_cape/ngram_dataset_family_104/api_descriptions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from common import load_json


# ---------------------------------------------------------------------------
# Curated Windows API descriptions (MSDN-based)
# Covers the most common APIs seen in malware analysis.
# ---------------------------------------------------------------------------

API_DESCRIPTIONS: Dict[str, str] = {
    # Process / Thread
    "CreateProcessInternalW": "Create a new process and its primary thread with extended internal options",
    "CreateProcessA": "Create a new process and its primary thread using ANSI path",
    "CreateProcessW": "Create a new process and its primary thread using Unicode path",
    "OpenProcess": "Open an existing local process object for access",
    "TerminateProcess": "Terminate the specified process and all of its threads",
    "ExitProcess": "End the calling process and all its threads",
    "CreateThread": "Create a thread to execute within the virtual address space of the calling process",
    "CreateRemoteThread": "Create a thread that runs in the virtual address space of another process",
    "CreateRemoteThreadEx": "Create a thread in another process with extended attributes",
    "ResumeThread": "Decrement the suspend count of a thread and resume execution",
    "SuspendThread": "Suspend the specified thread execution",
    "GetExitCodeThread": "Retrieve the termination status of a thread",
    "GetExitCodeProcess": "Retrieve the termination status of a process",
    "SetThreadContext": "Set the context including registers of a specified thread",
    "GetThreadContext": "Retrieve the context of the specified thread",
    "QueueUserAPC": "Add a user-mode asynchronous procedure call to the APC queue of a thread",
    "SwitchToThread": "Yield execution to another thread ready to run",
    "Sleep": "Suspend execution of the current thread for a specified interval",
    "SleepEx": "Suspend current thread execution with alertable wait option",
    "WaitForSingleObject": "Wait until the specified object is signaled or timeout",
    "WaitForMultipleObjects": "Wait until one or all of the specified objects are signaled",

    # Memory
    "VirtualAlloc": "Reserve commit or change the state of memory pages in the calling process",
    "VirtualAllocEx": "Reserve commit or change memory pages in the address space of another process",
    "VirtualFree": "Release or decommit pages of memory in the calling process",
    "VirtualFreeEx": "Release or decommit pages of memory in another process",
    "VirtualProtect": "Change the protection on a region of committed pages in the calling process",
    "VirtualProtectEx": "Change the protection on memory pages in another process",
    "VirtualQuery": "Query information about pages in the virtual address space of the calling process",
    "VirtualQueryEx": "Query information about pages in the virtual address space of another process",
    "ReadProcessMemory": "Read data from an area of memory in another process",
    "WriteProcessMemory": "Write data to an area of memory in another process",
    "HeapCreate": "Create a private heap object for the calling process",
    "HeapAlloc": "Allocate a block of memory from a heap",
    "HeapFree": "Free a memory block allocated from a heap",
    "GlobalAlloc": "Allocate global memory for data storage",
    "GlobalFree": "Free global memory allocation",
    "LocalAlloc": "Allocate local memory for data storage",
    "LocalFree": "Free local memory allocation",

    # Native API - Process / Memory
    "NtAllocateVirtualMemory": "Native API to allocate virtual memory in a process",
    "NtFreeVirtualMemory": "Native API to free virtual memory in a process",
    "NtProtectVirtualMemory": "Native API to change memory protection attributes",
    "NtReadVirtualMemory": "Native API to read virtual memory of another process",
    "NtWriteVirtualMemory": "Native API to write to virtual memory of another process",
    "NtCreateSection": "Native API to create a section object for memory mapping",
    "NtMapViewOfSection": "Native API to map a view of a section into process address space",
    "NtUnmapViewOfSection": "Native API to unmap a view of a section from process address space",
    "NtCreateProcess": "Native API to create a new process",
    "NtCreateProcessEx": "Native API to create a new process with extended parameters",
    "NtCreateUserProcess": "Native API to create a new user-mode process",
    "NtCreateThreadEx": "Native API to create a new thread with extended parameters",
    "NtResumeThread": "Native API to resume a suspended thread",
    "NtSuspendThread": "Native API to suspend a thread",
    "NtTerminateProcess": "Native API to terminate a process",
    "NtQueueApcThread": "Native API to queue an APC to a thread",
    "NtSetContextThread": "Native API to set thread execution context",
    "NtQueryInformationProcess": "Native API to query process information",
    "NtQuerySystemInformation": "Native API to query system-wide information",
    "NtDelayExecution": "Native API to delay thread execution for a specified interval",
    "NtQueryVirtualMemory": "Native API to query virtual memory region information",
    "RtlCreateUserThread": "Runtime library function to create a user-mode thread",
    "RtlWriteProcessMemory": "Runtime library function to write to process memory",

    # File I/O
    "CreateFileA": "Create or open a file or device using ANSI name",
    "CreateFileW": "Create or open a file or device using Unicode name",
    "ReadFile": "Read data from a file or input device",
    "WriteFile": "Write data to a file or output device",
    "DeleteFileA": "Delete an existing file using ANSI path",
    "DeleteFileW": "Delete an existing file using Unicode path",
    "CopyFileA": "Copy an existing file to a new file using ANSI path",
    "CopyFileW": "Copy an existing file to a new file using Unicode path",
    "CopyFileExW": "Copy a file with extended options and progress callback",
    "MoveFileA": "Move or rename a file using ANSI path",
    "MoveFileW": "Move or rename a file using Unicode path",
    "MoveFileExA": "Move or rename a file with extended options",
    "MoveFileExW": "Move or rename a file with extended options using Unicode path",
    "GetFileSize": "Retrieve the size of a file in bytes",
    "GetFileSizeEx": "Retrieve the size of a file as a 64-bit value",
    "SetFilePointer": "Move the file pointer of an open file",
    "SetFilePointerEx": "Move the file pointer with 64-bit offset support",
    "SetEndOfFile": "Set the end-of-file position for a file",
    "GetFileAttributesA": "Retrieve file system attributes for a file using ANSI path",
    "GetFileAttributesW": "Retrieve file system attributes for a file using Unicode path",
    "GetFileAttributesExW": "Retrieve extended file system attributes",
    "SetFileAttributesW": "Set the attributes for a file or directory",
    "FindFirstFileA": "Search a directory for the first file matching a pattern",
    "FindFirstFileW": "Search a directory for the first file matching a Unicode pattern",
    "FindFirstFileExA": "Search a directory with extended filtering options",
    "FindFirstFileExW": "Search a directory with extended filtering using Unicode",
    "FindNextFileA": "Continue searching for files matching a pattern",
    "FindNextFileW": "Continue searching for files matching a Unicode pattern",
    "FindClose": "Close a file search handle",
    "GetFileType": "Determine the type of a file handle",
    "CreateDirectoryW": "Create a new directory using Unicode path",
    "CreateDirectoryExW": "Create a new directory with a template directory",
    "RemoveDirectoryA": "Remove an existing empty directory",
    "RemoveDirectoryW": "Remove an existing empty directory using Unicode path",
    "GetTempPathW": "Retrieve the path of the temporary file directory",
    "GetTempFileNameW": "Create a name for a temporary file",
    "NtCreateFile": "Native API to create or open a file or device",
    "NtOpenFile": "Native API to open an existing file",
    "NtReadFile": "Native API to read data from a file",
    "NtWriteFile": "Native API to write data to a file",
    "NtDeleteFile": "Native API to delete a file",
    "NtQueryAttributesFile": "Native API to query file attributes",
    "NtQueryDirectoryFile": "Native API to enumerate directory contents",
    "NtSetInformationFile": "Native API to set file information",
    "NtQueryInformationFile": "Native API to query file information",
    "NtDeviceIoControlFile": "Native API to send device I/O control code",

    # Registry
    "RegOpenKeyExA": "Open a registry key for access using ANSI name",
    "RegOpenKeyExW": "Open a registry key for access using Unicode name",
    "RegCreateKeyExA": "Create or open a registry key using ANSI name",
    "RegCreateKeyExW": "Create or open a registry key using Unicode name",
    "RegSetValueExA": "Set data for a registry value using ANSI name",
    "RegSetValueExW": "Set data for a registry value using Unicode name",
    "RegQueryValueExA": "Query data for a registry value using ANSI name",
    "RegQueryValueExW": "Query data for a registry value using Unicode name",
    "RegDeleteKeyA": "Delete a registry subkey using ANSI name",
    "RegDeleteKeyW": "Delete a registry subkey using Unicode name",
    "RegDeleteValueA": "Delete a registry value using ANSI name",
    "RegDeleteValueW": "Delete a registry value using Unicode name",
    "RegEnumKeyExA": "Enumerate subkeys of an open registry key",
    "RegEnumKeyExW": "Enumerate subkeys of an open registry key using Unicode",
    "RegEnumValueA": "Enumerate the values for an open registry key",
    "RegEnumValueW": "Enumerate the values for an open registry key using Unicode",
    "RegCloseKey": "Close a handle to a registry key",
    "RegQueryInfoKeyA": "Retrieve information about a registry key",
    "RegQueryInfoKeyW": "Retrieve information about a registry key using Unicode",
    "NtOpenKey": "Native API to open a registry key",
    "NtOpenKeyEx": "Native API to open a registry key with extended options",
    "NtCreateKey": "Native API to create or open a registry key",
    "NtSetValueKey": "Native API to set a registry value",
    "NtQueryValueKey": "Native API to query a registry value",
    "NtDeleteKey": "Native API to delete a registry key",
    "NtDeleteValueKey": "Native API to delete a registry value",
    "NtEnumerateKey": "Native API to enumerate registry subkeys",
    "NtEnumerateValueKey": "Native API to enumerate registry values",
    "NtQueryKey": "Native API to query registry key information",

    # Network - WinINet / WinHTTP
    "InternetOpenA": "Initialize WinINet library for HTTP communications",
    "InternetOpenW": "Initialize WinINet library for HTTP communications using Unicode",
    "InternetConnectA": "Open HTTP or FTP connection to a server",
    "InternetConnectW": "Open HTTP or FTP connection to a server using Unicode",
    "InternetOpenUrlA": "Open a resource specified by a URL",
    "InternetOpenUrlW": "Open a resource specified by a Unicode URL",
    "HttpOpenRequestA": "Create an HTTP request handle for a connection",
    "HttpOpenRequestW": "Create an HTTP request handle using Unicode",
    "HttpSendRequestA": "Send an HTTP request to the server",
    "HttpSendRequestW": "Send an HTTP request using Unicode headers",
    "HttpQueryInfoA": "Query information about an HTTP request or response",
    "InternetReadFile": "Read data from an open Internet handle",
    "InternetWriteFile": "Write data to an open Internet handle",
    "InternetCloseHandle": "Close an Internet handle",
    "InternetCrackUrlA": "Parse a URL into its component parts",
    "InternetCrackUrlW": "Parse a URL into its component parts using Unicode",
    "InternetSetOptionA": "Set an Internet option for a handle",
    "WinHttpOpen": "Initialize WinHTTP library for HTTP communications",
    "WinHttpConnect": "Open WinHTTP connection to a server",
    "WinHttpOpenRequest": "Create a WinHTTP request handle",
    "WinHttpSendRequest": "Send an HTTP request via WinHTTP",
    "WinHttpReceiveResponse": "Wait for and receive an HTTP response via WinHTTP",
    "WinHttpReadData": "Read response data from WinHTTP handle",
    "WinHttpQueryHeaders": "Query HTTP response headers via WinHTTP",
    "URLDownloadToFileW": "Download a file from a URL and save to local filesystem",
    "URLDownloadToCacheFileW": "Download a file from URL to Internet cache",

    # Network - Sockets
    "WSAStartup": "Initialize Windows Sockets library",
    "socket": "Create a new network socket for communication",
    "connect": "Establish a connection to a remote socket",
    "bind": "Bind a socket to a local address and port",
    "listen": "Place a socket in listening state for incoming connections",
    "accept": "Accept an incoming connection on a listening socket",
    "send": "Send data on a connected socket",
    "recv": "Receive data from a connected socket",
    "sendto": "Send data to a specific destination address",
    "recvfrom": "Receive data and obtain the source address",
    "closesocket": "Close a socket and release associated resources",
    "select": "Monitor multiple sockets for readability or writability",
    "gethostbyname": "Resolve a hostname to an IP address",
    "getaddrinfo": "Resolve a hostname and service name to socket addresses",
    "WSASend": "Send data on a socket using overlapped I/O",
    "WSARecv": "Receive data on a socket using overlapped I/O",
    "WSASocketA": "Create a socket with extended attributes",
    "WSASocketW": "Create a socket with extended attributes using Unicode",

    # DNS
    "DnsQuery_A": "Query DNS for resource records using ANSI domain name",
    "DnsQuery_W": "Query DNS for resource records using Unicode domain name",

    # Network - Config
    "GetAdaptersInfo": "Retrieve network adapter information including IP addresses",
    "GetAdaptersAddresses": "Retrieve network adapter addresses and configuration",
    "GetNetworkParams": "Retrieve network parameters including DNS servers",
    "GetIpAddrTable": "Retrieve the IP address table for all network interfaces",

    # Dynamic Loading
    "LoadLibraryA": "Load a dynamic-link library module into the process",
    "LoadLibraryW": "Load a DLL using Unicode library name",
    "LoadLibraryExA": "Load a DLL with extended load options",
    "LoadLibraryExW": "Load a DLL with extended options using Unicode name",
    "GetModuleHandleA": "Get handle to a loaded module using ANSI name",
    "GetModuleHandleW": "Get handle to a loaded module using Unicode name",
    "GetModuleHandleExW": "Get handle to a loaded module with extended options",
    "GetProcAddress": "Retrieve the address of an exported function from a loaded DLL",
    "FreeLibrary": "Free a loaded DLL module from the process",
    "LdrLoadDll": "Native loader function to load a DLL module",
    "LdrGetProcedureAddress": "Native loader function to get exported function address",
    "LdrGetDllHandle": "Native loader function to get DLL module handle",
    "LdrUnloadDll": "Native loader function to unload a DLL module",

    # Process Enumeration
    "CreateToolhelp32Snapshot": "Take a snapshot of processes threads and modules in the system",
    "Process32First": "Retrieve information about the first process in a snapshot",
    "Process32FirstW": "Retrieve first process information using Unicode",
    "Process32Next": "Retrieve information about the next process in a snapshot",
    "Process32NextW": "Retrieve next process information using Unicode",
    "Module32First": "Retrieve information about the first module of a process",
    "Module32FirstW": "Retrieve first module information using Unicode",
    "Module32Next": "Retrieve information about the next module of a process",
    "Module32NextW": "Retrieve next module information using Unicode",
    "Thread32First": "Retrieve information about the first thread in a snapshot",
    "Thread32Next": "Retrieve information about the next thread in a snapshot",
    "EnumProcesses": "Enumerate the process identifiers for all processes in the system",

    # Service Management
    "OpenSCManagerA": "Open a connection to the Service Control Manager",
    "OpenSCManagerW": "Open Service Control Manager connection using Unicode",
    "OpenServiceA": "Open a handle to an existing service",
    "OpenServiceW": "Open a service handle using Unicode name",
    "CreateServiceA": "Create a new Windows service",
    "CreateServiceW": "Create a new Windows service using Unicode",
    "StartServiceA": "Start a Windows service",
    "StartServiceW": "Start a Windows service using Unicode name",
    "ControlService": "Send a control code to a service",
    "DeleteService": "Mark a service for deletion from the service database",
    "EnumServicesStatusA": "Enumerate services in the service database",
    "EnumServicesStatusW": "Enumerate services using Unicode",
    "EnumServicesStatusExA": "Enumerate services with extended status information",
    "EnumServicesStatusExW": "Enumerate services with extended status using Unicode",
    "QueryServiceStatusEx": "Query the status of a service with extended information",
    "ChangeServiceConfigA": "Change the configuration of a service",
    "ChangeServiceConfigW": "Change service configuration using Unicode",
    "ChangeServiceConfig2A": "Change optional configuration parameters of a service",
    "ChangeServiceConfig2W": "Change optional service configuration using Unicode",
    "CloseServiceHandle": "Close a handle to a service or the SCM",

    # Security / Tokens
    "OpenProcessToken": "Open the access token associated with a process",
    "OpenThreadToken": "Open the access token associated with a thread",
    "DuplicateTokenEx": "Create a new token that duplicates an existing token",
    "AdjustTokenPrivileges": "Enable or disable privileges in an access token",
    "GetTokenInformation": "Retrieve information about an access token",
    "ImpersonateLoggedOnUser": "Impersonate the security context of a logged-on user",
    "LookupAccountNameA": "Look up the security identifier for an account name",
    "LookupAccountNameW": "Look up SID for an account name using Unicode",
    "LookupAccountSidA": "Look up the account name for a security identifier",
    "LookupAccountSidW": "Look up account name for a SID using Unicode",
    "LookupPrivilegeValueA": "Look up the LUID for a privilege name",
    "LookupPrivilegeValueW": "Look up privilege LUID using Unicode name",
    "NtOpenProcessToken": "Native API to open a process access token",
    "NtAdjustPrivilegesToken": "Native API to adjust token privileges",

    # Crypto
    "CryptAcquireContextA": "Acquire a handle to a cryptographic service provider",
    "CryptAcquireContextW": "Acquire CSP handle using Unicode provider name",
    "CryptCreateHash": "Create a hash object for data hashing",
    "CryptHashData": "Add data to a hash object",
    "CryptEncrypt": "Encrypt data using a cryptographic key",
    "CryptDecrypt": "Decrypt data using a cryptographic key",
    "CryptGenKey": "Generate a random cryptographic key",
    "CryptDeriveKey": "Derive a cryptographic key from a hash value",
    "CryptImportKey": "Import a cryptographic key from a key blob",
    "CryptExportKey": "Export a cryptographic key to a key blob",
    "CryptDestroyKey": "Destroy a cryptographic key handle",
    "CryptDestroyHash": "Destroy a hash object",
    "CryptReleaseContext": "Release a cryptographic service provider handle",
    "CryptUnprotectData": "Decrypt and verify data protected by CryptProtectData using DPAPI",
    "CryptProtectData": "Encrypt data using Windows Data Protection API",
    "BCryptEncrypt": "Encrypt data using BCrypt algorithm",
    "BCryptDecrypt": "Decrypt data using BCrypt algorithm",
    "BCryptOpenAlgorithmProvider": "Open a BCrypt algorithm provider",
    "BCryptGenerateSymmetricKey": "Generate a symmetric key using BCrypt",

    # Credential Access
    "CredReadA": "Read a credential from the Windows credential store",
    "CredReadW": "Read a credential from credential store using Unicode",
    "CredWriteA": "Write a credential to the Windows credential store",
    "CredWriteW": "Write a credential to credential store using Unicode",
    "CredEnumerateA": "Enumerate credentials in the credential store",
    "CredEnumerateW": "Enumerate credentials using Unicode",
    "CredDeleteA": "Delete a credential from the credential store",
    "CredDeleteW": "Delete a credential using Unicode",
    "MiniDumpWriteDump": "Write a minidump of a process memory to a file",

    # Screen / Input
    "BitBlt": "Transfer a block of pixel data between device contexts for screen capture",
    "PrintWindow": "Copy a visual window image into a device context",
    "GetDC": "Retrieve a handle to a device context for a window or screen",
    "GetWindowDC": "Retrieve a device context for the entire window area",
    "ReleaseDC": "Release a device context obtained by GetDC",
    "CreateDIBSection": "Create a device-independent bitmap for direct pixel manipulation",
    "GetAsyncKeyState": "Determine if a key is pressed or was pressed since last check",
    "GetKeyState": "Retrieve the status of a virtual key",
    "SetWindowsHookExA": "Install a hook procedure to monitor system events",
    "SetWindowsHookExW": "Install a hook procedure using Unicode",
    "UnhookWindowsHookEx": "Remove a hook procedure installed by SetWindowsHookEx",
    "GetRawInputData": "Retrieve raw input data from an input device",
    "RegisterRawInputDevices": "Register devices that supply raw input",

    # Clipboard
    "OpenClipboard": "Open the clipboard for examination or modification",
    "GetClipboardData": "Retrieve data from the clipboard in a specified format",
    "SetClipboardData": "Place data on the clipboard in a specified format",
    "EmptyClipboard": "Empty the clipboard and free associated data",
    "CloseClipboard": "Close the clipboard after access",

    # System Info
    "GetSystemInfo": "Retrieve system hardware and processor information",
    "GetNativeSystemInfo": "Retrieve system information for the native processor architecture",
    "GetVersionExA": "Retrieve Windows operating system version information",
    "GetVersionExW": "Retrieve OS version information using Unicode",
    "RtlGetVersion": "Native function to retrieve OS version information",
    "GetSystemMetrics": "Retrieve various system metrics and configuration settings",
    "GetComputerNameA": "Retrieve the NetBIOS name of the local computer",
    "GetComputerNameW": "Retrieve computer name using Unicode",
    "GetComputerNameExW": "Retrieve computer name in extended format",
    "GetUserNameA": "Retrieve the user name of the current thread",
    "GetUserNameW": "Retrieve user name using Unicode",
    "GetUserNameExW": "Retrieve user name in extended format",
    "GlobalMemoryStatusEx": "Retrieve information about physical and virtual memory usage",
    "GetDiskFreeSpaceExW": "Retrieve free disk space information for a volume",
    "GetDiskFreeSpaceW": "Retrieve disk space information for a drive",
    "GetVolumeInformationW": "Retrieve file system and volume information",

    # Time
    "GetLocalTime": "Retrieve the current local date and time",
    "GetSystemTime": "Retrieve the current UTC system date and time",
    "GetTickCount": "Retrieve the number of milliseconds since system start",
    "GetTickCount64": "Retrieve milliseconds since system start as 64-bit value",
    "QueryPerformanceCounter": "Retrieve high-resolution performance counter value",
    "QueryPerformanceFrequency": "Retrieve the frequency of the performance counter",
    "GetSystemTimeAsFileTime": "Retrieve current system time as a FILETIME structure",

    # COM
    "CoCreateInstance": "Create a single COM object of a specified class",
    "CoInitializeEx": "Initialize the COM library for use by the calling thread",
    "CoInitialize": "Initialize the COM library on the current thread",
    "CoUninitialize": "Close the COM library on the current thread",
    "CoGetClassObject": "Retrieve a COM class factory for a specified class",

    # Shell
    "ShellExecuteExW": "Perform an operation on a file with extended options",
    "ShellExecuteA": "Perform an operation on a file such as open or print",
    "ShellExecuteW": "Perform a file operation using Unicode parameters",
    "SHGetFolderPathW": "Retrieve the path of a known system folder",
    "SHGetSpecialFolderPathW": "Retrieve the path of a special system folder",

    # Window / Message
    "CreateWindowExA": "Create a window with extended style attributes",
    "CreateWindowExW": "Create a window with extended style using Unicode",
    "FindWindowA": "Find a top-level window by class name or title",
    "FindWindowW": "Find a window using Unicode class name or title",
    "FindWindowExA": "Find a child window by class name and title",
    "FindWindowExW": "Find a child window using Unicode parameters",
    "GetMessageA": "Retrieve a message from the calling thread message queue",
    "GetMessageW": "Retrieve a message using Unicode",
    "PeekMessageA": "Check for a message in the queue without removing it",
    "PeekMessageW": "Check message queue using Unicode",
    "PostMessageA": "Post a message to the message queue of a window",
    "PostMessageW": "Post a message using Unicode",
    "SendMessageA": "Send a message to a window and wait for processing",
    "SendMessageW": "Send a message using Unicode and wait for processing",
    "SendNotifyMessageA": "Send a message and return without waiting for processing",
    "SendNotifyMessageW": "Send a notification message using Unicode",
    "SetWindowLongA": "Change an attribute of a window",
    "SetWindowLongW": "Change a window attribute using Unicode",
    "SetWindowLongPtrA": "Change a window attribute with pointer-size value",
    "SetWindowLongPtrW": "Change a window attribute with pointer-size value Unicode",
    "ShowWindow": "Set the specified window show state",
    "DestroyWindow": "Destroy a window and its child windows",

    # Anti-Debug / Evasion
    "IsDebuggerPresent": "Determine whether the calling process is being debugged",
    "CheckRemoteDebuggerPresent": "Determine whether a process is being debugged by a remote debugger",
    "OutputDebugStringA": "Send a string to the debugger for display",
    "OutputDebugStringW": "Send a Unicode string to the debugger",

    # Misc
    "GetLastError": "Retrieve the last error code for the calling thread",
    "SetLastError": "Set the last error code for the calling thread",
    "GetCurrentProcessId": "Retrieve the process identifier of the calling process",
    "GetCurrentThreadId": "Retrieve the thread identifier of the calling thread",
    "GetCurrentProcess": "Retrieve a pseudo handle for the current process",
    "CloseHandle": "Close an open object handle",
    "DuplicateHandle": "Duplicate an object handle from one process to another",
    "DeviceIoControl": "Send a device I/O control code to a device driver",
    "GetModuleFileNameA": "Retrieve the fully qualified path of a loaded module",
    "GetModuleFileNameW": "Retrieve module path using Unicode",
    "GetModuleFileNameExW": "Retrieve module path for a specified process",
    "GetCommandLineA": "Retrieve the command line string for the current process",
    "GetCommandLineW": "Retrieve the command line using Unicode",
    "GetEnvironmentVariableA": "Retrieve the value of an environment variable",
    "GetEnvironmentVariableW": "Retrieve environment variable using Unicode",
    "SetEnvironmentVariableA": "Set the value of an environment variable",
    "SetEnvironmentVariableW": "Set environment variable using Unicode",
    "GetStartupInfoW": "Retrieve startup information for the current process",
    "GetProcessHeap": "Retrieve a handle to the default heap of the calling process",
    "FlushFileBuffers": "Flush all buffered data for a file to the storage device",
    "GetFileInformationByHandle": "Retrieve file information for an open file handle",
    "GetFileInformationByHandleEx": "Retrieve extended file information by handle",
    "SetFileInformationByHandle": "Set file information for an open file handle",
    "GetFullPathNameW": "Retrieve the full path and file name of a specified file",
    "SearchPathW": "Search for a file in a specified path",
}


def generate_fallback(api_name: str) -> str:
    """Generate a basic description by splitting the API name into words."""
    # Remove A/W/Ex suffixes
    name = api_name
    for suffix in ("ExW", "ExA", "Ex", "W", "A"):
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            name = name[:-len(suffix)]
            break

    # Split camelCase
    words = []
    current = []
    for ch in name:
        if ch.isupper() and current:
            words.append("".join(current))
            current = [ch.lower()]
        else:
            current.append(ch.lower())
    if current:
        words.append("".join(current))

    return " ".join(words) + " API function"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Windows API descriptions for desc features."
    )
    parser.add_argument(
        "--api-list",
        type=Path,
        default=Path("data/avast_ctu_cape/ngram_dataset_family_104/api.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: same dir as api-list / api_descriptions.json).",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        default=True,
        help="Match API names case-insensitively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_list.exists():
        print(f"[ERROR] API list not found: {args.api_list}")
        return

    api_list: List[str] = load_json(args.api_list)
    output_path = args.output or (args.api_list.parent / "api_descriptions.json")

    # Build lookup (case-insensitive)
    lookup: Dict[str, str] = {}
    if args.case_insensitive:
        for k, v in API_DESCRIPTIONS.items():
            lookup[k.lower()] = v
    else:
        lookup = dict(API_DESCRIPTIONS)

    # Match APIs
    descriptions: Dict[str, str] = {}
    matched = 0
    fallback = 0

    for api in api_list:
        key = api.lower() if args.case_insensitive else api
        if key in lookup:
            descriptions[api] = lookup[key]
            matched += 1
        else:
            descriptions[api] = generate_fallback(api)
            fallback += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    total = len(api_list)
    print(f"[INFO] API list:    {total} APIs")
    print(f"[INFO] Matched:     {matched} ({100*matched/total:.1f}%)")
    print(f"[INFO] Fallback:    {fallback} ({100*fallback/total:.1f}%)")
    print(f"[INFO] Output:      {output_path}")


if __name__ == "__main__":
    main()
