$input = "D:\car-detect-output\20240425\D2\*"
$output = "C:\Users\AI Training\Documents\lamp_detection\test"
$exclude_model = "PRIUS 5DR HATCHBACK (AUTO)"

$make_models = Get-ChildItem $input -Recurse -Directory -Exclude $exclude_model |
  Measure-Object |
  Select -Expand Count

$makes = Get-ChildItem $input -Directory -Exclude $exclude_model | Measure-Object | Select -Expand Count

$models = $make_models - $makes

Get-ChildItem -Path $input -Recurse -Directory -Exclude $exclude_model |
    ForEach-Object {
        Copy-Item -Path $_.GetFiles()[0].FullName -Destination $output
    }

$total_images = Get-ChildItem $output -File | Measure-Object | Select -Expand Count

Write-Output "Expected number of images: $($models)"
Write-Output "Actual number of images: $($total_images)"
