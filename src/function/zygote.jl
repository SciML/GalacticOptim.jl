struct AutoZygote <: AbstractADType end

function instantiate_function(f, x, ::AutoZygote, p, num_cons = 0)
    num_cons != 0 && error("AutoZygote does not currently support constraints")

    _f = (θ, args...) -> f(θ,p,args...)[1]
    if f.grad === nothing
        grad = (res, θ, args...) -> res isa DiffResults.DiffResult ? DiffResults.gradient!(res, Zygote.gradient(x -> _f(x, args...), θ)[1]) : res .= Zygote.gradient(x -> _f(x, args...), θ)[1]
    else
        grad = f.grad
    end

    if f.hess === nothing
        hess = function (res, θ, args...)
            if res isa DiffResults.DiffResult
                DiffResults.hessian!(res, ForwardDiff.jacobian(θ) do θ
                                                Zygote.gradient(x -> _f(x, args...), θ)[1]
                                            end)
            else
                res .=  ForwardDiff.jacobian(θ) do θ
                    Zygote.gradient(x ->_f(x, args...), θ)[1]
                end
            end
        end
    else
        hess = f.hess
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            _θ = ForwardDiff.Dual.(θ, v)
            res = DiffResults.GradientResult(_θ)
            grad(res, _θ, args...)
            H .= getindex.(ForwardDiff.partials.(DiffResults.gradient(res)),1)
        end
    else
        hv = f.hv
    end

    return OptimizationFunction{false,AutoZygote,typeof(f),typeof(grad),typeof(hess),typeof(hv),Nothing,Nothing,Nothing}(f,AutoZygote(),grad,hess,hv,nothing,nothing,nothing)
end
