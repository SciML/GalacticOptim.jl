import MathOptInterface
const MOI = MathOptInterface

struct MOIOptimizationProblem{T,F<:OptimizationFunction,uType} <: MOI.AbstractNLPEvaluator
    f::F
    u0::uType
    J::Matrix{T}
    H::Matrix{T}
    cons_H::Vector{Matrix{T}}
end

MOI.eval_objective(moiproblem::MOIOptimizationProblem, x) = moiproblem.f.f(x)

MOI.eval_constraint(moiproblem::MOIOptimizationProblem, g, x) = g .= moiproblem.f.cons(x)

MOI.eval_objective_gradient(moiproblem::MOIOptimizationProblem, G, x) = moiproblem.f.grad(G, x)

function MOI.eval_hessian_lagrangian(moiproblem::MOIOptimizationProblem{T}, h, x, σ, μ) where {T}
    n = length(moiproblem.u0)
    a = zeros(n, n)
    moiproblem.f.hess(a, x)
    if iszero(σ)
        fill!(h, zero(T))
    else
        moiproblem.f.hess(moiproblem.H, x)
        k = 0
        for col in 1:n
            for row in 1:col
                k += 1
                h[k] = σ * moiproblem.H[row, col]
            end
        end
    end
    if !isempty(μ) && !all(iszero, μ)
        moiproblem.f.cons_h(moiproblem.cons_H, x)
        for (μi, Hi) in zip(μ, moiproblem.cons_H)
            k = 0
            for col in 1:n
                for row in 1:col
                    k += 1
                    h[k] += μi * Hi[row, col]
                end
            end
        end
    end
    return
end

function MOI.eval_constraint_jacobian(moiproblem::MOIOptimizationProblem, j, x)
    isempty(j) && return
    moiproblem.f.cons_j === nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")
    n = length(moiproblem.u0)
    moiproblem.f.cons_j(moiproblem.J, x)
    for i in eachindex(j)
        j[i] = moiproblem.J[i]
    end
end

function MOI.jacobian_structure(moiproblem::MOIOptimizationProblem)
    return Tuple{Int,Int}[(con, var) for con in 1:size(moiproblem.J,1) for var in 1:size(moiproblem.J,2)]
end

function MOI.hessian_lagrangian_structure(moiproblem::MOIOptimizationProblem)
    return Tuple{Int,Int}[(row, col) for col in 1:length(moiproblem.u0) for row in 1:col]
end

function MOI.initialize(moiproblem::MOIOptimizationProblem, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MOI.features_available(moiproblem))
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
end

MOI.features_available(moiproblem::MOIOptimizationProblem) = [:Grad, :Hess, :Jac]

function make_moi_problem(prob::OptimizationProblem)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p,num_cons)
    T = eltype(prob.u0)
    n = length(prob.u0)
    moiproblem = MOIOptimizationProblem(f,prob.u0,zeros(T,num_cons,n),zeros(T,n,n),Matrix{T}[zeros(T,n,n) for i in 1:num_cons])
    return moiproblem
end

function __solve(prob::OptimizationProblem, opt::Union{Function, Type{<:MOI.AbstractOptimizer}, MOI.OptimizerWithAttributes}; sense = MinSense)
    optimizer = MOI.instantiate(opt)
    num_variables = length(prob.u0)
	θ = MOI.add_variables(optimizer, num_variables)
	if prob.lb !== nothing
        @assert eachindex(prob.lb) == Base.OneTo(num_variables)
		for i in 1:num_variables
			MOI.add_constraint(optimizer, MOI.SingleVariable(θ[i]), MOI.GreaterThan(prob.lb[i]))
        end
    end
	if prob.ub !== nothing
        @assert eachindex(prob.ub) == Base.OneTo(num_variables)
		for i in 1:num_variables
			MOI.add_constraint(optimizer, MOI.SingleVariable(θ[i]), MOI.LessThan(prob.ub[i]))
        end
    end
    @assert eachindex(prob.u0) == Base.OneTo(num_variables)
	for i in 1:num_variables
		MOI.set(optimizer, MOI.VariablePrimalStart(), θ[i], prob.u0[i])
	end
    MOI.set(optimizer, MOI.ObjectiveSense(), sense === MinSense ? MOI.MIN_SENSE : MOI.MAX_SENSE)
    if prob.lcons === nothing
        @assert prob.ucons === nothing
        con_bounds = MOI.NLPBoundsPair[]
    else
        @assert prob.ucons !== nothing
        con_bounds = MOI.NLPBoundsPair.(prob.lcons, prob.ucons)
    end
	MOI.set(optimizer, MOI.NLPBlock(), MOI.NLPBlockData(con_bounds, make_moi_problem(prob), true))
	MOI.optimize!(optimizer)
    if MOI.get(optimizer, MOI.ResultCount()) >= 1
        minimizer = MOI.get(optimizer, MOI.VariablePrimal(), θ)
        minimum = MOI.get(optimizer, MOI.ObjectiveValue())
    else
        minimizer = fill(NaN, num_variables)
        minimum = NaN
    end
    SciMLBase.build_solution(prob, opt, minimizer, minimum; original=nothing)
end
