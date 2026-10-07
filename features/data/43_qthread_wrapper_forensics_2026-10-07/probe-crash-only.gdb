set pagination off
set auto-load safe-path /dev/null
set debuginfod enabled off
run
info registers rbp rbx r15 rdi rsi rip rsp
printf "SBK_WRAPPER %p C_PTR_ARRAY %p INDEX %ld\n", $rbp, $r15, $rbx
x/6gx $rbp
set $pytype = *(void**)($rbp+8)
printf "PYTYPE %p\n", $pytype
x/5gx $pytype
x/s *(char**)($pytype+24)
x/4gx $r15
bt 20
