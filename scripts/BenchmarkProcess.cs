using System;
using System.Collections;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

// One benchmark launch owns one Windows job. Create suspended so no child runs
// outside that job, including launchers which immediately start a replacement.
public sealed class BenchmarkProcess : IDisposable
{
    private IntPtr job;
    public Process Process { get; private set; }

    private BenchmarkProcess(IntPtr ownedJob, Process root)
    {
        job = ownedJob;
        Process = root;
    }

    public uint ActiveProcesses
    {
        get
        {
            if (job == IntPtr.Zero) return 0;
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION info;
            if (!QueryInformationJobObject(job, 1, out info,
                (uint)Marshal.SizeOf(typeof(JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION)), IntPtr.Zero))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return info.ActiveProcesses;
        }
    }

    public void Dispose()
    {
        IntPtr ownedJob = job;
        job = IntPtr.Zero;
        if (ownedJob != IntPtr.Zero) CloseHandle(ownedJob);
        if (Process != null) Process.Dispose();
    }

    public static BenchmarkProcess Start(string executable, string[] arguments,
        string workingDirectory, IDictionary environment, string stdoutPath, string stderrPath)
    {
        if (String.IsNullOrEmpty(executable) || String.IsNullOrEmpty(workingDirectory) ||
            String.IsNullOrEmpty(stdoutPath) || String.IsNullOrEmpty(stderrPath) ||
            arguments == null || environment == null) throw new ArgumentException("Complete launch inputs required.");
        IntPtr ownedJob = IntPtr.Zero, stdout = IntPtr.Zero, stderr = IntPtr.Zero, stdin = IntPtr.Zero;
        PROCESS_INFORMATION child = new PROCESS_INFORMATION();
        Process root = null;
        IntPtr environmentBlock = IntPtr.Zero;
        bool resumed = false;
        try
        {
            ownedJob = CreateJobObject(IntPtr.Zero, null);
            CheckHandle(ownedJob);
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION limits = new JOB_OBJECT_EXTENDED_LIMIT_INFORMATION();
            limits.BasicLimitInformation.LimitFlags = 0x00002000; // KILL_ON_JOB_CLOSE
            if (!SetInformationJobObject(ownedJob, 9, ref limits,
                (uint)Marshal.SizeOf(typeof(JOB_OBJECT_EXTENDED_LIMIT_INFORMATION))))
                throw new Win32Exception(Marshal.GetLastWin32Error());

            SECURITY_ATTRIBUTES security = new SECURITY_ATTRIBUTES();
            security.nLength = (uint)Marshal.SizeOf(typeof(SECURITY_ATTRIBUTES));
            security.bInheritHandle = true;
            stdout = CreateFile(stdoutPath, 0x40000000, 1, ref security, 1, 0x80, IntPtr.Zero);
            CheckHandle(stdout);
            stderr = CreateFile(stderrPath, 0x40000000, 1, ref security, 1, 0x80, IntPtr.Zero);
            CheckHandle(stderr);
            stdin = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0x80, IntPtr.Zero);
            CheckHandle(stdin);

            STARTUPINFO startup = new STARTUPINFO();
            startup.cb = (uint)Marshal.SizeOf(typeof(STARTUPINFO));
            startup.dwFlags = 0x00000100; // STARTF_USESTDHANDLES
            startup.hStdInput = stdin;
            startup.hStdOutput = stdout;
            startup.hStdError = stderr;
            environmentBlock = Marshal.StringToHGlobalUni(BuildEnvironment(environment));
            StringBuilder command = new StringBuilder(Quote(executable));
            foreach (string argument in arguments) command.Append(' ').Append(Quote(argument));
            if (!CreateProcess(executable, command, IntPtr.Zero, IntPtr.Zero, true,
                0x00000004 | 0x00000400 | 0x08000000, environmentBlock, workingDirectory,
                ref startup, out child))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            if (!AssignProcessToJobObject(ownedJob, child.hProcess))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            root = Process.GetProcessById((int)child.dwProcessId);
            if (root.Handle == IntPtr.Zero) throw new InvalidOperationException("Root process handle unavailable.");
            if (ResumeThread(child.hThread) == UInt32.MaxValue)
                throw new Win32Exception(Marshal.GetLastWin32Error());
            resumed = true;
            BenchmarkProcess owner = new BenchmarkProcess(ownedJob, root);
            ownedJob = IntPtr.Zero;
            root = null;
            return owner;
        }
        finally
        {
            if (!resumed && child.hProcess != IntPtr.Zero) TerminateProcess(child.hProcess, 1);
            if (root != null) root.Dispose();
            CloseIfValid(child.hThread); CloseIfValid(child.hProcess);
            CloseIfValid(stdout); CloseIfValid(stderr); CloseIfValid(stdin);
            if (environmentBlock != IntPtr.Zero) Marshal.FreeHGlobal(environmentBlock);
            CloseIfValid(ownedJob);
        }
    }

    private static string BuildEnvironment(IDictionary variables)
    {
        var entries = new System.Collections.Generic.SortedDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (DictionaryEntry entry in variables)
        {
            string key = entry.Key as string;
            string value = entry.Value == null ? null : Convert.ToString(entry.Value, System.Globalization.CultureInfo.InvariantCulture);
            if (String.IsNullOrEmpty(key) || key.IndexOf('=') >= 0 || key.IndexOf('\0') >= 0 ||
                value == null || value.IndexOf('\0') >= 0)
                throw new ArgumentException("Invalid environment entry: " + key);
            entries.Add(key, value);
        }
        StringBuilder block = new StringBuilder();
        foreach (var entry in entries) block.Append(entry.Key).Append('=').Append(entry.Value).Append('\0');
        if (entries.Count == 0) block.Append('\0');
        block.Append('\0');
        return block.ToString();
    }

    private static string Quote(string value)
    {
        if (value == null || value.IndexOf('\0') >= 0) throw new ArgumentException("Invalid argument.");
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char ch in value)
        {
            if (ch == '\\') { slashes++; continue; }
            if (ch == '"') { result.Append('\\', slashes * 2 + 1); result.Append('"'); }
            else { result.Append('\\', slashes); result.Append(ch); }
            slashes = 0;
        }
        result.Append('\\', slashes * 2).Append('"');
        return result.ToString();
    }

    private static void CheckHandle(IntPtr handle)
    {
        if (handle == IntPtr.Zero || handle == new IntPtr(-1))
            throw new Win32Exception(Marshal.GetLastWin32Error());
    }
    private static void CloseIfValid(IntPtr handle)
    {
        if (handle != IntPtr.Zero && handle != new IntPtr(-1)) CloseHandle(handle);
    }

    [StructLayout(LayoutKind.Sequential)] private struct SECURITY_ATTRIBUTES
    { public uint nLength; public IntPtr lpSecurityDescriptor; [MarshalAs(UnmanagedType.Bool)] public bool bInheritHandle; }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] private struct STARTUPINFO
    {
        public uint cb; public string lpReserved, lpDesktop, lpTitle;
        public uint dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
        public ushort wShowWindow, cbReserved2; public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError;
    }
    [StructLayout(LayoutKind.Sequential)] private struct PROCESS_INFORMATION
    { public IntPtr hProcess, hThread; public uint dwProcessId, dwThreadId; }
    [StructLayout(LayoutKind.Sequential)] private struct JOB_OBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit; public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize; public uint ActiveProcessLimit;
        public UIntPtr Affinity; public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)] private struct IO_COUNTERS
    { public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount; }
    [StructLayout(LayoutKind.Sequential)] private struct JOB_OBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOB_OBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }
    [StructLayout(LayoutKind.Sequential)] private struct JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION
    {
        public long TotalUserTime, TotalKernelTime, ThisPeriodTotalUserTime, ThisPeriodTotalKernelTime;
        public uint TotalPageFaultCount, TotalProcesses, ActiveProcesses, TotalTerminatedProcesses;
    }
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref JOB_OBJECT_EXTENDED_LIMIT_INFORMATION info, uint length);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool QueryInformationJobObject(IntPtr job, int infoClass, out JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION info, uint length, IntPtr returnLength);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] private static extern IntPtr CreateFile(string path, uint access, uint share, ref SECURITY_ATTRIBUTES attributes, uint disposition, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)] private static extern bool CreateProcess(string application, StringBuilder command, IntPtr processAttributes, IntPtr threadAttributes, [MarshalAs(UnmanagedType.Bool)] bool inheritHandles, uint flags, IntPtr environment, string workingDirectory, ref STARTUPINFO startup, out PROCESS_INFORMATION process);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool TerminateProcess(IntPtr process, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool CloseHandle(IntPtr handle);
}
