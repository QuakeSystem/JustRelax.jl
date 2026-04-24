using Infiltrator
function main()
    # do test addition and subtraction
    a = 1.0
    b = 2.0
    c = a + b
    d = a - b
    @infiltrate
    println("a: ", a)
    println("b: ", b)
    println("a + b: ", c)
    println("a - b: ", d)
end
main()