# Find IP Address
Write-Output "Finding IP address..."
$ip_list = Get-NetIPAddress -AddressFamily IPv4 | 
    Where-Object { 
        $_.IPAddress -match '^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$' -and  # IPv4 format filtering
        $_.IPAddress -notmatch '^169\.' -and                             # Excluding IP starting with 169.
        $_.IPAddress -notmatch '^127\.'                                  # Excluding IP starting with 127.
    } | Select-Object -ExpandProperty IPAddress

# If there is no filtered IP
if (-not $ip_list -or $ip_list.Count -eq 0) {
    Write-Output "IPv4 address not found."
    exit 1
}

# Output IP list
Write-Output "`nList of available IP addresses:"
for ($i = 0; $i -lt $ip_list.Count; $i++) {
    Write-Output "$($i+1): $($ip_list[$i])"
}

# Get user input
do {
    $selection = Read-Host "Select the number of IP address to use (1-$($ip_list.Count))"
} while ($selection -notmatch '^\d+$' -or [int]$selection -lt 1 -or [int]$selection -gt $ip_list.Count)

# Apply selected IP
$user_ip = $ip_list[[int]$selection - 1]
Write-Output "Selected IP address: $user_ip"

# Running the Nomad client installation script in WSL
Write-Output "Install Nomad Client on WSL"
wsl /mnt/c/temp/nomad_client_setup.sh $user_ip