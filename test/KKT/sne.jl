@testset "SNE" begin

    A = SparseMatrixCSC{Float64, Int}([
        1 0 1 0;
        0 1 0 1
    ])

    @testset "SNE1" begin
        kkt = KKT.SNE(A, variant=false)
        KKT.run_ls_tests(A, kkt)
    end
    
    @testset "SNE2" begin
        kkt = KKT.SNE(A, variant=true)
        KKT.run_ls_tests(A, kkt)
    end

end
