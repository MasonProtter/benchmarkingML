using DelimitedFiles, Statistics

# Load grid for log(y) and transition matrix
const logy_grid = readdlm("logy_grid.txt")[:]
const Py = readdlm("P.txt")


function main(nB=351, repeats=500)
    β = .953
    γ = 2.
    r = 0.017
    θ = 0.282
    ny = size(logy_grid, 1)

    Bgrid = LinRange(-.45, .45, nB)
    ygrid = exp.(logy_grid)

    ymean = mean(ygrid) .+ 0 * ygrid
    def_y = min(0.969 * ymean, ygrid)

    Vd = zeros(ny, 1)
    Vc = zeros(ny, nB)
    V = zeros(ny, nB)
    Q = ones(ny, nB) * .95

    y = reshape(ygrid, (ny, 1, 1))
    B = reshape(Bgrid, (1, nB, 1))
    Bnext = reshape(Bgrid, (1, 1, nB))

    zero_ind = Int(ceil(nB / 2))

    u(c, γ) = c^(1 - γ) / (1 - γ)

    t0 = time()
    function iterate(V, Vc, Vd, Q)
        EV = Py * V
        EVd = Py * Vd
        EVc = Py * Vc

        Vd_target = u.(def_y, γ) + β * (θ * EVc[:, zero_ind] + (1 - θ) * EVd[:])
        Vd_target = reshape(Vd_target, (ny, 1))

        Qnext = reshape(Q, (ny, 1, nB))

        m = Array{Float64,3}(undef, ny, nB, nB)
        @inbounds for k = 1:nB
            Bk = Bgrid[k]
            for j = 1:nB
                Bj = Bgrid[j]
                @simd for i = 1:ny
                    c = ygrid[i] - Q[i,k] * Bk + Bj
                    c = max(c, 1e-14)
                    m[i, j, k] = u(c, γ) + β * EV[i,j]
                end
            end
        end
        # c = @. y - Qnext * Bnext + B
        # map!(x->max(x, 1e-14), c, c)
        # EV = reshape(EV, (ny, 1, nB))
        # m =  @. u(c, γ) + β * EV
        Vc_target = reshape(maximum(m, dims=3), (ny, nB))

        Vd_compat = Vd * ones(1, nB)
        default_states = float(Vd_compat .> Vc)
        default_prob = Py * default_states
        Q_target = (1. .- default_prob) ./ (1 + r)

        V_target = max.(Vc, Vd_compat)

        return V_target, Vc_target, Vd_target, Q_target

    end

    iterate(V, Vc, Vd, Q)  # warmup
    t0 = time()
    for iteration in 1:repeats
        V, Vc, Vd, Q = iterate(V, Vc, Vd, Q)
    end
    t1 = time()
    out = (t1 - t0) / repeats

end

print(1000 * main(1351, 10))
