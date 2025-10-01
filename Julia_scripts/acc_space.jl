global dir = ARGS[end]
cd(dir) # Change to the specified directory

using PyPlot #v2.11.6
using CSV 
using DataFrames
using StatsBase
using Printf
using NCDatasets #v0.10.4

const cm = 1/2.54 #centimeter in inches
PyPlot.ioff() #desliga processo interativo
    
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

function read_data(fpath::String, nxnz::Tuple; veloc::Bool=false, surface::Bool=false)
    var = split(fpath, '_')
    file = "$(var[1])/$(var[1])_$(var[2]).txt"
    C = CSV.File(file, header=false, comment="P", skipto=3,types=Float32)|>CSV.Tables.matrix
    
    Nx,Nz = nxnz
    if veloc 
        C[C .< 1e-200] .= 0
        vx = transpose(reshape(C[1:2:end], (Nx, Nz)))
        vy = transpose(reshape(C[2:2:end], (Nx, Nz)))
        R = (vx,vy)
        return R
    
    elseif surface
        return (C[1:end,1],C[1:end,2]) #sx, sy
    
    else
        C[C .< 1e-200] .= 0
        R = transpose(reshape(C[1:end], (Nx, Nz)))
        return R
    end
end

function read_time(step::Int)::Float32
    time::Float32=0.0
    open(joinpath("time", "time_$step.txt")) do file
        line = readline(file)
        line = split(line,"   ")[2]
        time = parse(Float32,line)/1e6 #Myr
    return time
    end
end


param = read_param("param.txt")
global step_initial::Int = 0
global Nx::Int = parse(Int,param["nx"])
global Nz::Int = parse(Int,param["nz"])
global Lx::Float32 = parse(Float32,param["lx"])
global Lz::Float32 = parse(Float32,param["lz"])
global d_step::Int = parse(Int,param["step_print"])
global step_final::Int = parse(Int,ARGS[1])
global h_air::Float32 = 40.0e3 #km
levels = Int32[0,-500,-1000,-2000,-2500,-3000]

#finding the accomadation space and the topographic indices
global times = Vector{Float32}(undef,Int(step_final/d_step)+1)
global acc_spaces = Dict{Int32,Vector{Float32}}()
for level in levels
    acc_spaces[level] = Vector{Float32}(undef,Int(step_final/d_step)+1) #Cria um vetor cheio de indefinido para cada level
end

#global Acc_space = Vector{Float32}(undef,Int(step_final/d_step)+1)
#global mean_submerse = Vector{Float32}(undef,Int(step_final/d_step)+1)

@inbounds for (i,step) in enumerate(step_initial:d_step:step_final)
    #read time
    time::Float32=read_time(step)
    #read and fix the surface

    sx,sy = read_data("surface_$step",(Nx,Nz),surface=true)
    sy .= sy .+ h_air
    fix_topo = mean(sy[(sx.>100e3).&(sx.< 250e3)]) #m
    sy .= sy .- fix_topo

    #print("step $step, time = $time Myr ||")
    #println("max = $(maximum(sy)) m || min = $(minimum(sy)) m || fix = $(fix_topo)" )

    #plot surface
    fig,ax = subplots(figsize=(18*cm,6*cm))
    ax.plot(sx./1e3,sy,"k-")
    ax.set_title("$(round(time,digits=2)) Myr")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(0,Lx/1000)
    ax.set_ylim(-5500,3000)

    fig.tight_layout()
    fname=@sprintf("topography_%05d.png", step)
    savefig(fname,dpi=150)
    close(fig)

    times[i] = time
    @inbounds @simd for level in levels
        acc_spaces[level][i] = sum(abs.(sy[sy.<level]))*(sx[2]-sx[1]) #m²
    end
    #Acc_space[i] = sum(abs.(sy[sy.<0]))*(sx[2]-sx[1]) #m²
end

colors = ["blue","royalblue","cornflowerblue","deepskyblue","turquoise","green"]
figure(figsize=(20*cm,20*cm))
for (j,level) in enumerate(levels)
    plot(times, acc_spaces[level]./1e6, color=colors[j], ls="-", label="$level m")
end
#plot(times, acc_spaces/1e6, color="k", ls="-")
#plot(times, times.*25, color="red", ls="--")

#axhline(500,color="blue",ls="--",label="500 km²")

grid(true)
legend()
xlim(0,maximum(times))
ylim(0)
xlabel("Time [Myr]")
ylabel("Accumulated Space [km²]")
savefig("acc_space.png",dpi=150)
