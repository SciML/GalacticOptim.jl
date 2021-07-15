"""
$(DocStringExtensions.README)
"""
module GalacticOptim

using DocStringExtensions
using Reexport
@reexport using SciMLBase
using Requires
using DiffResults
using Logging, ProgressLogging, Printf, ConsoleProgressMonitor, TerminalLoggers, LoggingExtras
using ArrayInterface, Base.Iterators

import SciMLBase: OptimizationProblem, OptimizationFunction, AbstractADType, __solve

@enum ObjSense MinSense MaxSense

include("utils.jl")
include("function/function.jl")

function __init__()
    # Optimization backends
    @require BlackBoxOptim="a134a8b2-14d6-55f6-9291-3336d3ab0209" include("solve/blackboxoptim.jl")
    @require CMAEvolutionStrategy="8d3b24bd-414e-49e0-94fb-163cc3a3e411" include("solve/cmaevolutionstrategy.jl")
    @require Evolutionary="86b6b26d-c046-49b6-aa0b-5f0f74682bd6" include("solve/evolutionary.jl")
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("solve/flux.jl")
    @require MathOptInterface="b8f27783-ece8-5eb3-8dc8-9495eed66fee" include("solve/moi.jl")
    @require MultistartOptimization="3933049c-43be-478e-a8bb-6e0f7fd53575" include("solve/multistartoptimization.jl")
    @require NLopt="76087f3c-5699-56af-9a33-bf431cd00edd" include("solve/nlopt.jl")
    @require Optim="429524aa-4258-5aef-a3af-852621145aeb" include("solve/optim.jl")
    @require QuadDIRECT="dae52e8d-d666-5120-a592-9e15c33b8d7a" include("solve/quaddirect.jl")

    # AD backends
    @require FiniteDiff="6a86dc24-6348-571c-b903-95158fe2bd41" include("function/finitediff.jl")
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("function/forwarddiff.jl")
    @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("function/reversediff.jl")
    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("function/tracker.jl")
    @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" include("function/zygote.jl")
end

export solve

end # module
