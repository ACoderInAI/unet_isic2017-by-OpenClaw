# System Information Report
Write-Host "=== COMPUTER SYSTEM INFORMATION ===" -ForegroundColor Cyan
Write-Host ""

# CPU
Write-Host "--- CPU Information ---" -ForegroundColor Yellow
$cpu = Get-WmiObject Win32_Processor | Select-Object -First 1
Write-Host "Processor: $($cpu.Name)"
Write-Host "Cores: $($cpu.NumberOfCores)"
Write-Host "Logical Processors: $($cpu.NumberOfLogicalProcessors)"
Write-Host "Max Clock Speed: $($cpu.MaxClockSpeed) MHz"
Write-Host ""

# GPU
Write-Host "--- GPU Information ---" -ForegroundColor Yellow
$gpu = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"} | Select-Object -First 1
if ($gpu) {
    $ramGB = [math]::Round($gpu.AdapterRAM / 1GB, 2)
    Write-Host "GPU: $($gpu.Name)"
    Write-Host "VRAM: $ramGB GB ($($gpu.AdapterRAM) bytes)"
    Write-Host "Driver Version: $($gpu.DriverVersion)"
} else {
    Write-Host "No NVIDIA GPU found"
}
Write-Host ""

# RAM
Write-Host "--- RAM Information ---" -ForegroundColor Yellow
$ram = Get-WmiObject Win32_ComputerSystem
$ramGB = [math]::Round($ram.TotalPhysicalMemory / 1GB, 2)
Write-Host "Total Physical Memory: $ramGB GB"
Write-Host ""

# Storage
Write-Host "--- Storage Disks ---" -ForegroundColor Yellow
$disks = Get-WmiObject Win32_LogicalDisk -Filter "DriveType=3"
foreach ($disk in $disks) {
    $sizeGB = [math]::Round($disk.Size / 1GB, 2)
    $freeGB = [math]::Round($disk.FreeSpace / 1GB, 2)
    $usedGB = [math]::Round(($disk.Size - $disk.FreeSpace) / 1GB, 2)
    Write-Host "Drive $($disk.DeviceID) ($($disk.VolumeName)):"
    Write-Host "  Total: $sizeGB GB"
    Write-Host "  Used:  $usedGB GB"
    Write-Host "  Free:  $freeGB GB"
}
Write-Host ""

# OS
Write-Host "--- Operating System ---" -ForegroundColor Yellow
$os = Get-WmiObject Win32_OperatingSystem
Write-Host "OS: $($os.Caption)"
Write-Host "Version: $($os.Version)"
Write-Host "Architecture: $($os.OSArchitecture)"
Write-Host ""

Write-Host "=== END OF REPORT ===" -ForegroundColor Cyan
