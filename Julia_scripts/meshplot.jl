#changing the directory
global dir = ARGS[end]
cd(dir)

#iniciando os workers
using BenchmarkTools
using Distributed
if nprocs() == 1
    addprocs(Threads.nthreads()-1) # adiciona workers baseado no número de núcleos da CPU
end

@everywhere begin
using Glob
using CSV
using DataFrames
using PyPlot
using Printf

const cm = 1/2.54 #centimeter in inches
PyPlot.ioff() #desliga processo interativo
end

print("Number of workers: ")
println(nprocs())

@everywhere begin
function read_param(fpath::String="param.txt")::Dict{String,String}

    param_dict = Dict{String}{String}()
    open(fpath, "r") do file
        for line in eachline(file)
            line=strip(line)
            if isempty(line) || startswith(line, "#")
                continue
            end
            line = split(line, "#")[1] # Remove comments
            line = replace(line, " " => "")
            key_value = split(line, "=")
            param_dict[lowercase(key_value[1])] = key_value[2]
        end
    end
    return param_dict
end
#=
function find_nearest_positive(mat::Matrix{Int16}, i::Int16, j::Int16, radius::Int = 1)
    rows::Int16, cols::Int16 = size(mat)
    neighbors = [
        (i-radius, j), (i+radius, j), (i, j-radius), (i, j+radius),
        (i-radius, j-radius), (i-radius, j+radius), (i+radius, j-radius), (i+radius, j+radius)
    ]
    
    r = Int16[]
    for n in neighbors
        ni, nj = n
        
        if ni < 1 || ni > rows || nj < 1 || nj > cols
            continue # Skip out-of-bounds neighbors
        
        elseif mat[ni, nj] >= 0
            push!(r, mat[ni, nj])
        end

    end
    
    return r

    #counts = countmap(r)
    #v,_ = findmax(counts) #qual o é o valor mais frequente
    
    
    #return Int16(v)
end
=#

function replace_negatives_with_neighbors!(mat::Matrix{Int16})
    #result::Matrix{Int16} = copy(mat)
    result = mat #faz referencia ao mesmo local na memória
    rows::Int16, cols::Int16 = size(mat)

    negative_indices::Vector{CartesianIndex{2}} = findall(result .< 0)

    #println("okok - ", typeof(result))
    @inbounds for idx in negative_indices #negative address
        i::Int16, j::Int16 = idx[1], idx[2]

        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ni, nj = i + di, j + dj

            # Verifica se o vizinho está dentro dos limites e se seu valor
            # na matriz original não é negativo.
            if (1 <= ni <= rows && 1 <= nj <= cols) && mat[ni, nj] >= 0
                result[i, j] = mat[ni, nj]
                break # Para a busca assim que um vizinho válido é encontrado.
            end
        end
    end
    return result
end

function read_litho_file(fpath::String)
    # Read lithology file and return x, z, lith
    CLit = CSV.read(fpath, DataFrame, header=false, comment="P", delim=" ")
    return (CLit[1:end,1], CLit[1:end,2], CLit[1:end,3])
end

function read_data(fpath::String, nxnz::Tuple, veloc::Bool=false, surface::Bool=false)
    var = split(fpath, '_')
    file = "$(var[1])/$(var[1])_$(var[2]).txt"
    C = CSV.File(file, header=false, comment="P", skipto=3,types=Float32)|>CSV.Tables.matrix
    C[C .< 1e-200] .= 0

    Nx,Nz = nxnz
    if veloc 
        
        vx = transpose(reshape(C[1:2:end], (Nx, Nz)))
        vy = transpose(reshape(C[2:2:end], (Nx, Nz)))
        C = (vx,vy)
        return C
    
    elseif surface
        return C
    
    else
        C = transpose(reshape(C[1:end], (Nx, Nz)))
        return C
    end
end

function plot_litho(step::Int, pdic::Dict{Any,Any})
    #read dict params
    litho_dict = pdic["litho_dict"]
    Nx = pdic["Nx"]
    Nz = pdic["Nz"]
    Lx = pdic["Lx"]
    Lz = pdic["Lz"]
    ncores = pdic["ncores"]
    mycolors = pdic["mycolors"]
    dir = pdic["dir"]
    temp_intervals = pdic["temp_intervals"]

    xi = LinRange(0, Lx/1000, Nx) #km
    zi = LinRange(0, Lz/1000, Nz) #km
    Nxl = (Nx-1)*5
    Nzl = (Nz-1)*5

    #read time
    time::Float32=0.0
    open(joinpath("time", "time_$step.txt")) do file
        line = readline(file)
        line = split(line,"   ")[2]
        time = parse(Float32,line)/1e6 #Myr
    end

    #create litho mesh
    litho_mesh = zeros(Int16,Nzl,Nxl) .- Int16(1)
    println("step: $step")

    @inbounds for core in 0:(ncores-1)
        fpath=joinpath("litho","litho_$step"*"_$core.txt")
        x_core, z_core, litho_core = read_litho_file(fpath)
        @inbounds @simd for i in eachindex(x_core)
            x_idx = x_core[i] + 1
            z_idx = z_core[i] + 1
            litho_mesh[z_idx, x_idx] = litho_core[i]
        end        
    end

    #@. muda o array original (equivalente a .= .+)
    # = .+ cria um novo array 

    println("ok - lithos lidos")
    #inbounds = Remove verificações de limites (bounds checking)
    #simd = "Single Instruction Multiple Data" Habilita vetorização do loop pelo compilador


    litho_mesh = replace_negatives_with_neighbors!(litho_mesh)
    litho_mesh .= get.(Ref(litho_dict), litho_mesh, litho_mesh) #dictionary operation
    println("ok - lithos processados")

    fig, ax = plt.subplots(figsize=(20*cm,5*cm), dpi=300)
    ax.imshow(litho_mesh, extent=[0, Lx/1000, (Lz/1000)-40, -40],  
                      cmap=mycolors,vmin=0,vmax=7, aspect="auto")
    litho_mesh = nothing  #libera memória

    #ploting strain
    density = read_data("density_$step", (Nx,Nz))
    strain = read_data("strain_$step", (Nx,Nz))
    strain[density .< 1000] .= 0
    density = nothing #libera memória
    @. strain = log10(strain)
    ax.imshow(strain, extent=[0, Lx/1000,-40, (Lz/1000)-40], 
    cmap="Greys", alpha=0.2, vmin=-0.5, vmax=0.9, aspect="auto")
    strain = nothing #libera memória

    #ploting temperature contours
    if size(temp_intervals)[1] >= 1
        println("Ploting temperature contours")
        temp = read_data("temperature_$step", (Nx,Nz))
        ax.contour(xi, reverse(zi).-40, temp, 
        levels=temp_intervals, colors="r", linewidths=0.5)
        temp = nothing
    end

    #cbar = colorbar(mesh, ticks=0:7)
    #cbar.ax.set_yticklabels(["Ast", "Lit", "LC", "UC", "Sed1", "Sed2", "Salt", "Air"])

    # -- personalizar o grafico --
    #ax.invert_yaxis()
    ax.set_xlabel("x [km]")
    ax.set_ylabel("Depth [km]")
    ax.set_xlim(0, Lx/1000)
    ax.set_ylim(180,-10)
    ax.annotate("$(round(time,digits=2)) Myr",(0.2,1.02), xycoords="axes fraction", fontsize=8, color="k")
    

    fig.tight_layout()
    fname = @sprintf("litho_%05d.png", step)
    savefig(joinpath(dir,fname))
    println("Saved: $fname")

    PyPlot.close(fig)
    return nothing
end

end

cr::Int = 255 #white

color_air  = Float16[1,1,1]
color_salt = Float16[242/cr, 97/cr, 128/cr]
color_sed1 = Float16[241/cr, 184/cr, 68/cr]
color_sed2 = Float16[128/cr, 95/cr, 40/cr]
color_uc   = Float16[165/cr, 163/cr, 238/cr]
color_lc   = Float16[227/cr, 217/cr, 242/cr]
color_lit  = Float16[155/cr, 194/cr, 155/cr]
color_ast  = Float16[207/cr, 226/cr, 205/cr]

colors = Vector{Float16}[color_ast,color_lit,color_lc,color_uc,color_sed1,color_sed2,color_salt,color_air]
#layers = collect(0:length(colors)) ./ length(colors) # todas as camadas


my_cmap = PyPlot.matplotlib.colors.ListedColormap(colors,"my_cmap")

global litho_dict = Dict{Int8,Int}(
    2 => 1,  # Lithospheric mantle
    3 => 1,  # Lithospheric mantle
    4 => 2,  # Lower crust
    5 => 3,  # Upper crust
    6 => 4,  # Sed 1
    7 => 5,  # Sed 2
    8 => 6,  # Salt
    9 => 7   # air
)


param = read_param() 
#Tudo que é atribuido dentro de um escopo (função, loop, etc) é local por padrão
#Para modificar uma variável global dentro de um escopo, é necessário usar a palavra-chave 'global'
global step_initial::Int = 0
global Nx::Int = parse(Int,param["nx"])
global Nz::Int = parse(Int,param["nz"])
global Lx::Float32 = parse(Float32,param["lx"])
global Lz::Float32 = parse(Float32,param["lz"])
global d_step::Int = parse(Int,param["step_print"])
global step_final::Int = parse(Int,ARGS[1])
name::String = split(basename(glob(joinpath("litho","litho_0_*.txt"))[end]),'.')
ncores::Int = parse(Int,split(name,'_')[end])+1

global temp_intervals = Int32[600, 800, 1000, 1200, 1350] #°C

println("step_initial: $step_initial, step_final: $step_final, d_step: $d_step, ncores: $ncores")
println("Nx: $Nx, Nz: $Nz, Lx: $Lx, Lz: $Lz")

pdic = Dict{Any,Any}(
    "litho_dict" => litho_dict, "dir" => dir,
    "Nx" => Nx,"Nz" => Nz,"Lx" => Lx,"Lz" => Lz,
    "ncores" => ncores, 
    "mycolors" => my_cmap,
    "temp_intervals" => temp_intervals,

)

try
@sync @distributed for step in step_initial:d_step:step_final
    println("Processing step: $step")
    plot_litho(step, pdic)
    println("Finished step: $step")

end
finally
    rmprocs(workers()) #remove os workers
    println("All workers removed.")
end

#pmap(step -> plot_litho(step, pdic), step_initial:d_step:step_final)