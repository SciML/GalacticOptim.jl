using SafeTestsets, Pkg, Hyperopt, Optim

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end
@time begin
if GROUP == "All" || GROUP == "Core"
    @safetestset "Rosenbrock" begin include("rosenbrock.jl") end
    @safetestset "AD Tests" begin include("ADtests.jl") end
    @safetestset "Mini batching" begin include("minibatch.jl") end
    @safetestset "DiffEqFlux" begin include("diffeqfluxtests.jl") end
end

if !is_APPVEYOR && GROUP == "Downstream"
    activate_downstream_env()
    Pkg.test("DiffEqFlux")
end

if !is_APPVEYOR && GROUP == "GPU"
    activate_downstream_env()
    @safetestset "DiffEqFlux GPU" begin include("downstream/gpu_neural_ode.jl") end
end
end
