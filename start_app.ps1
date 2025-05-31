# Auto-set API key from .env and launch app
$apiKey = (Get-Content .env | Where-Object {$_ -like "OPENAI_API_KEY=*"}) -replace "OPENAI_API_KEY=", ""
$adminPass = (Get-Content .env | Where-Object {$_ -like "ADMIN_PASSWORD=*"}) -replace "ADMIN_PASSWORD=", ""
$env:OPENAI_API_KEY = $apiKey
$env:ADMIN_PASSWORD = $adminPass
Write-Host " Starting Headline Hunter with API key: $($apiKey.Substring(0,20))..."
streamlit run streamlit_app.py
