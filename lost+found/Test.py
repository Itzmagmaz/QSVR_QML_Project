from classiq import *


@qfunc
def main(reg: Output[QArray]):
    allocate(6, reg)
    H(reg[0])
    n = reg.len
    repeat(
        reg.len - 1,
        lambda x: CX(reg[x], reg[x + 1])
    )



qprog_task = synthesize(main)
show(qprog_task)