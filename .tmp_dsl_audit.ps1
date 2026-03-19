$path='dataset/data/splits/Nhi/test.json'
$data=Get-Content -Raw $path | ConvertFrom-Json
$stats=[ordered]@{}
$samples=[ordered]@{}
function Add-Issue($key,$img,$detail){
  if(-not $stats.Contains($key)){ $stats[$key]=0; $samples[$key]=@() }
  $stats[$key]++
  if($samples[$key].Count -lt 6){ $samples[$key] += [pscustomobject]@{image=$img; detail=$detail} }
}

foreach($item in $data){
  $img=$item.image_dir
  $instr=$item.instruction
  $ans=$item.answer
  $lines=@()
  foreach($ln in ($ans -split "`n")){ $t=$ln.Trim(); if($t){ $lines += $t } }

  $angleDecl=[regex]::Matches($instr,'([A-Z]{3})\s*=\s*([0-9]+)')
  foreach($m in $angleDecl){
    $ang=$m.Groups[1].Value; $deg=$m.Groups[2].Value
    $x=$ang.Substring(0,1); $y=$ang.Substring(1,1); $z=$ang.Substring(2,1)
    $expected="(angle-measure $x $y $z $deg)"
    if($lines -notcontains $expected){ Add-Issue 'missing_or_wrong_angle_measure' $img $expected }
  }

  $defs=@{}
  foreach($ln in $lines){
    if($ln -match '^\(define\s+([A-Z])\s+point(\s|\))'){
      $p=$matches[1]
      if($defs.ContainsKey($p)){ Add-Issue 'duplicate_define_same_point' $img "define $p" } else { $defs[$p]=1 }
    }
  }

  foreach($ln in $lines){
    if($ln -match '^\(define\s+([A-Z])\s+point\s+\(on-segment\s+[^\)]+\)\)$'){ Add-Issue 'define_uses_on-segment_construction' $img $ln }
  }

  foreach($ln in $lines){
    if($ln -match '^\(on-segment\s+([^\)]+)\)$'){
      $parts=($matches[1] -split '\s+' | Where-Object {$_})
      if($parts.Count -ne 3){ Add-Issue 'on-segment_wrong_arity' $img $ln }
    }
  }

  $segSeen=@{}
  foreach($ln in $lines){
    if($ln -match '^\(segment\s+([A-Z])\s+([A-Z])\)$'){
      $a=$matches[1]; $b=$matches[2]
      $k=([string]::Join('',(@($a,$b) | Sort-Object)))
      if($segSeen.ContainsKey($k)){ Add-Issue 'duplicate_segment' $img "$a$b" } else { $segSeen[$k]=1 }
    }
  }

  $segments=@{}
  foreach($ln in $lines){
    if($ln -match '^\(segment\s+([A-Z])\s+([A-Z])\)$'){
      $a=$matches[1]; $b=$matches[2]
      $segments["$a$b"]=1; $segments["$b$a"]=1
    }
  }
  foreach($ln in $lines){
    if($ln -match '^\(perpendicular\s+\(segment\s+([A-Z])\s+([A-Z])\)\s+\(segment\s+([A-Z])\s+([A-Z])\)\)$'){
      $a=$matches[1]; $b=$matches[2]; $c=$matches[3]; $d=$matches[4]
      if(-not $segments.ContainsKey("$a$b")){ Add-Issue 'perpendicular_without_segment_decl' $img "missing segment $a$b" }
      if(-not $segments.ContainsKey("$c$d")){ Add-Issue 'perpendicular_without_segment_decl' $img "missing segment $c$d" }
    }
  }

  $shapeVerts=@{}
  foreach($ln in $lines){
    if($ln -match '^\((triangle|square|rectangle|trapezoid|parallelogram|rhombus)\s+\(([A-Z](?:\s+[A-Z]){2,3})\)'){
      $verts=($matches[2] -split '\s+')
      foreach($v in $verts){ $shapeVerts[$v]=1 }
    }
  }
  foreach($ln in $lines){
    if($ln -match '^\(define\s+([A-Z])\s+point(\s|\))'){
      $p=$matches[1]
      if($shapeVerts.ContainsKey($p)){ Add-Issue 'define_shape_vertex_again' $img "define $p" }
    }
  }

  $hasOInInstr=($instr -match '(^|[^A-Z])O([^A-Z]|$)|\(O\)|tâm O|đường tròn O| tại O|OA|OB|OC|OD')
  $hasOInAns=($ans -match '(^|[^A-Z])O([^A-Z]|$)')
  if((-not $hasOInInstr) -and $hasOInAns){ Add-Issue 'unexpected_O_token_in_answer' $img 'O appears but instruction has no O' }

  if($instr -match 'đường tròn\s+([A-Z])\s+với\s+đường\s+kính'){
    $expectedCenter=$matches[1]
    $diaLine=($lines | Where-Object { $_ -match '^\(diameter\s+[A-Z]\s+[A-Z]\s+([A-Z])\)$' } | Select-Object -First 1)
    if($diaLine){
      $diaLine -match '^\(diameter\s+[A-Z]\s+[A-Z]\s+([A-Z])\)$' | Out-Null
      $center=$matches[1]
      if($center -ne $expectedCenter){ Add-Issue 'diameter_center_mismatch_with_instruction' $img "expected center $expectedCenter but got $center" }
    }
  }
}

"=== SUMMARY ==="
$stats.GetEnumerator() | Sort-Object Name | ForEach-Object { "{0}: {1}" -f $_.Key, $_.Value }
"=== SAMPLE DETAILS ==="
foreach($k in ($samples.Keys | Sort-Object)){
  "[$k]"
  $samples[$k] | ForEach-Object { " - {0} :: {1}" -f $_.image,$_.detail }
}
