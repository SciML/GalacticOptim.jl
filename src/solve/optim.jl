
decompose_trace(trace::Optim.OptimizationTrace) = last(trace)

function __solve(prob::OptimizationProblem, opt::Optim.AbstractOptimizer,
                 data = DEFAULT_DATA;
                 maxiters = nothing,
                 cb = (args...) -> (false),
                 progress = false,
                 kwargs...)
    local x, cur, state

    if data != DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
        cb_call = opt == Optim.NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cur, state = iterate(data, state)
        cb_call
    end

    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    !(opt isa Optim.ZerothOrderOptimizer) && f.grad === nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function(θ)
        x = f.f(θ, prob.p, cur...)
        return first(x)
    end

    fg! = function (G,θ)
        if G !== nothing
            f.grad(G, θ, cur...)
        end
        return _loss(θ)
    end

    if opt isa Optim.KrylovTrustRegion
        optim_f = Optim.TwiceDifferentiableHV(_loss, fg!, (H,θ,v) -> f.hv(H,θ,v,cur...), prob.u0)
    else
        optim_f = Optim.TwiceDifferentiable(_loss, (G, θ) -> f.grad(G, θ, cur...), fg!, (H,θ) -> f.hess(H,θ,cur...), prob.u0)
    end

    original = Optim.optimize(optim_f, prob.u0, opt,
                              !(isnothing(maxiters)) ?
                                Optim.Options(;extended_trace = true,
                                               callback = _cb,
                                               iterations = maxiters,
                                               kwargs...) :
                                Optim.Options(;extended_trace = true,
                                               callback = _cb, kwargs...))
    SciMLBase.build_solution(prob, opt, original.minimizer,
                             original.minimum; original=original)
end

function __solve(prob::OptimizationProblem, opt::Union{Optim.Fminbox,Optim.SAMIN},
                 data = DEFAULT_DATA;
                 maxiters = nothing,
                 cb = (args...) -> (false),
                 progress = false,
                 kwargs...)

    local x, cur, state

    if data != DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

    function _cb(trace)
        cb_call = !(opt isa Optim.SAMIN) && opt.method == Optim.NelderMead() ? cb(decompose_trace(trace).metadata["centroid"],x...) : cb(decompose_trace(trace).metadata["x"],x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        end
        cur, state = iterate(data, state)
        cb_call
    end

    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p)

    !(opt isa Optim.ZerothOrderOptimizer) && f.grad === nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function(θ)
        x = f.f(θ, prob.p, cur...)
        return first(x)
    end
    fg! = function (G,θ)
        if G !== nothing
            f.grad(G, θ, cur...)
        end

        return _loss(θ)
    end
    optim_f = Optim.OnceDifferentiable(_loss, (G, θ) -> f.grad(G, θ, cur...), fg!, prob.u0)

    original = Optim.optimize(optim_f, prob.lb, prob.ub, prob.u0, opt,
                              !(isnothing(maxiters)) ? Optim.Options(;
                              extended_trace = true, callback = _cb,
                              iterations = maxiters, kwargs...) :
                              Optim.Options(;extended_trace = true,
                              callback = _cb, kwargs...))
    SciMLBase.build_solution(prob, opt, original.minimizer,
                             original.minimum; original=original)
end


function __solve(prob::OptimizationProblem, opt::Optim.ConstrainedOptimizer,
                 data = DEFAULT_DATA;
                 maxiters = nothing,
                 cb = (args...) -> (false),
                 progress = false,
                 kwargs...)

    local x, cur, state

    if data != DEFAULT_DATA
        maxiters = length(data)
    end

    cur, state = iterate(data)

      function _cb(trace)
      cb_call = cb(decompose_trace(trace).metadata["x"],x...)
      if !(typeof(cb_call) <: Bool)
          error("The callback should return a boolean `halt` for whether to stop the optimization process.")
      end
      cur, state = iterate(data, state)
      cb_call
    end

    if !(isnothing(maxiters)) && maxiters <= 0.0
        error("The number of maxiters has to be a non-negative and non-zero number.")
    elseif !(isnothing(maxiters))
        maxiters = convert(Int, maxiters)
    end

    f = instantiate_function(prob.f,prob.u0,prob.f.adtype,prob.p,prob.ucons === nothing ? 0 : length(prob.ucons))

    f.cons_j ===nothing && error("Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends")

    _loss = function(θ)
        x = f.f(θ, prob.p, cur...)
        return x[1]
    end
    fg! = function (G,θ)
        if G !== nothing
            f.grad(G, θ, cur...)
        end
        return _loss(θ)
    end
    optim_f = Optim.TwiceDifferentiable(_loss, (G, θ) -> f.grad(G, θ, cur...), fg!, (H,θ) -> f.hess(H, θ, cur...), prob.u0)

    cons! = (res, θ) -> res .= f.cons(θ);

    cons_j! = function(J, x)
        f.cons_j(J, x)
    end

    cons_hl! = function (h, θ, λ)
        res = [similar(h) for i in 1:length(λ)]
        f.cons_h(res, θ)
        for i in 1:length(λ)
            h .+= λ[i]*res[i]
        end
    end

    lb = prob.lb === nothing ? [] : prob.lb
    ub = prob.ub === nothing ? [] : prob.ub
    optim_fc = Optim.TwiceDifferentiableConstraints(cons!, cons_j!, cons_hl!, lb, ub, prob.lcons, prob.ucons)

    original = Optim.optimize(optim_f, optim_fc, prob.u0, opt,
                              !(isnothing(maxiters)) ? Optim.Options(;
                                extended_trace = true, callback = _cb,
                                iterations = maxiters, kwargs...) :
                                Optim.Options(;extended_trace = true,
                                callback = _cb, kwargs...))
    SciMLBase.build_solution(prob, opt, original.minimizer,
                             original.minimum; original=original)
end



