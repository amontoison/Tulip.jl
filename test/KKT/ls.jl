@testset "LS" begin

    A = SparseMatrixCSC{Float64, Int}([
        1 0 1 0;
        0 1 0 1
    ])

    @testset "LS1" begin
        kkt = KKT.LS(A, variant=false)
        KKT.run_ls_tests(A, kkt)
    end
    
    @testset "LS2" begin
        kkt = KKT.LS(A, variant=true)
        KKT.run_ls_tests(A, kkt)
    end

end
