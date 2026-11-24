try {
  Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/api/expenses/scan' -Method POST -TimeoutSec 8 | Out-Null
  Write-Output "unexpected success"
} catch {
  $resp = $_.Exception.Response
  if ($resp) { Write-Output "HTTP $([int]$resp.StatusCode)" }
  Write-Output $_.ErrorDetails.Message
}
