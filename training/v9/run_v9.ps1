# run_v9.ps1 -Run A : keeps Windows awake (no idle sleep) while train_v9.py runs; log -> v9\runs_v9<Run>.log
param([string]$Run = "A")
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$py = 'C:\Github\drone_training\v8\.venv\Scripts\python.exe'
$log = "C:\Github\drone_training\v9\runs_v9$Run.log"
"== start $(Get-Date -Format s)" | Out-File $log -Encoding utf8
& $py 'C:\Github\THIS_IS_CAMERA_ANDAU\training\v9\train_v9.py' $Run 0 *>> $log
"== exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File $log -Append -Encoding utf8
[W.P]::SetThreadExecutionState(0x80000000) | Out-Null
