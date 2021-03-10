@testset "LN" begin

    A = SparseMatrixCSC{Float64, Int}([
        1 0 1 0;
        0 1 0 1
    ])

    @testset "LN1" begin
        kkt = KKT.LN(A, variant=false)
        KKT.run_ls_tests(A, kkt)
    end
    
    @testset "LN2" begin
        kkt = KKT.LN(A, variant=true)
        KKT.run_ls_tests(A, kkt)
    end

end
