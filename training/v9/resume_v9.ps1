Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState(0x80000001) | Out-Null
$py = 'C:\Github\drone_training\v8\.venv\Scripts\python.exe'
& $py -c "from ultralytics import YOLO; YOLO(r'C:\Github\drone_training\v9\runs\v9A\weights\last.pt').train(resume=True, workers=0)" *> 'C:\Github\drone_training\v9\runs_v9A_resume.log'
"== exit $LASTEXITCODE $(Get-Date -Format s)" | Out-File 'C:\Github\drone_training\v9\runs_v9A_resume.done' -Encoding utf8
[W.P]::SetThreadExecutionState(0x80000000) | Out-Null