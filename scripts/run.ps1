# List of satellite names
$satellites = @("Fengyun-2F", "Fengyun-2H", "Sentinel-3A", "CryoSat-2", "SARAL")

# Loop through each satellite name
foreach ($satellite in $satellites) {
    Write-Host "Processing satellite: $satellite"
    
    # Run the Python program with the current satellite name
    python main.py --mode analysis --satellite_name $satellite --verbose
    python main.py --mode train --satellite_name $satellite --verbose
    
    Write-Host "Finished processing satellite: $satellite"
    Write-Host "----------------------------------------"
}