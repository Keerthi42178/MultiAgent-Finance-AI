$base = 'http://127.0.0.1:8000/api'
$reg = Invoke-RestMethod -Uri "$base/auth/register" -Method Post -ContentType 'application/json' -Body (@{
  email = "smoke$((Get-Random))@test.com"; password = 'password123'; full_name = 'Smoke' } | ConvertTo-Json)
$h = @{ Authorization = "Bearer $($reg.access_token)" }
$msg = "My monthly salary is 90000. I spend 30000 monthly and already pay 10000 in EMIs. I want to purchase a car worth 8 lakh. Evaluate whether I can afford it, estimate a safe down payment and EMI budget, and recommend when I should purchase it."
$r = Invoke-RestMethod -Uri "$base/chat/message" -Method Post -ContentType 'application/json' -Headers $h -Body (@{ message = $msg } | ConvertTo-Json)
Write-Output "KIND: $($r.kind)  AGENT: $($r.agent)"
Write-Output "TASK: $($r.data.task.task_key)  STATUS: $($r.data.task.status)"
Write-Output "SUBTASKS: $(($r.data.subtasks | ForEach-Object { "$($_.assigned_agent):$($_.status)" }) -join ', ')"
Write-Output "TRACE EVENTS: $($r.data.trace.Count)"
$r.data.trace | ForEach-Object { Write-Output "  [$($_.actor)] $($_.action) :: $($_.detail)" }
Write-Output "--- ANSWER ---"
Write-Output $r.answer
