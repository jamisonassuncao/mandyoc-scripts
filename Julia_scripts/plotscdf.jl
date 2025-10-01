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

#=
function read_netcdf(variable::String; steps::Tuple{Int,Int}=(-1,-1), get_all::Bool=true)
    ds = NCDataset("$variable.nc","r")
    if get_all == false
        dsv = reverse(ds[variable][:,:,:]',dims=1)
        if steps[2] != -1
            return dsv[:,:,steps[1]:steps[2]] #return only the steps between steps[1] and steps[2]
        else
            return dsv[:,:,:] #return all steps
        end
    else
        return ds #return the entire dataset: all attributes and all steps
    end
    close(ds)
end
=#

function read_times()::Tuple{Vector{Int}, Vector{Float32}}
    columns_types = Dict{String,DataType}("step" => Int, "time_myr" => Float32)
    ts = CSV.File("times.csv",header=true,types=columns_types)|>CSV.Tables.matrix
    return (ts[:,1], ts[:,2])
end

function plot_surface(data::NCDataset,stepindex::Int,pdict::Dict; strain::Any=false)
    time = pdict["times"][stepindex]
    step = pdict["steps"][stepindex]
    h_air = pdict["h_air"]
    plots_param = pdict["plots_param"]
    variable = pdict["variable"]    
    fig,axs = plt.subplots(1,1,(figsize=(24*cm,8*cm)),sharex=true)

    x = data["x"][:]./1e3 #km
    surf = data[variable][:,stepindex] #m
    
    axs.plot(x,surf,ls="solid",color="k")
    axs.axhline(-500,ls="--",alpha=0.7,color="#3C62FA")
    axs.axhline(-1000,ls="--",alpha=0.7,color="#3DB9E6")
    axs.axhline(-2000,ls="--",alpha=0.7,color="#3DE6C5")
    axs.axhline(-2500,ls="--",alpha=0.7,color="#5A54EE")

    axs.set_xlabel("X (km)")
    axs.set_ylabel("Z (m)")
    axs.set_title("$(time) Myr - step_$(step)")

    axs.set_xlim(x[1],x[end])
    axs.set_ylim(plots_param["ylim"][1],plots_param["ylim"][2])
    axs.grid(plots_param["grid"])

    if strain!=false
        plt.pcolormesh(strain["x"][:]./1e3, reverse(strain["z"][:]) .*-1 .+ h_air, 
        strain["strain"][:,:,stepindex]',
            cmap="Reds",alpha=0.4,vmin=0.0,vmax=0.8)

    end
    fig.tight_layout()
    fname = Printf.@sprintf("surface_%05d.png", step)
    fig.savefig(fname, dpi=300)
    close(fig)
end

function acc_space_old(surface::Vector{Float32},x::Vector{Float32},baselevel::Int)::Float32
    acc::Float32 = -1.0
    dx::Float32 = x[2]-x[1]
    surface .= baselevel .- surface
    acc = sum( abs.(surface[surface .> 0]) )*dx #m²
    return acc
end

function acc_space(data::NCDataset,pdict::Dict)::Dict{Int,Vector{Float32}}
    all_s::Matrix = data["surface"][:,:] #todas superfícies
    dx::Float32 = data["x"][2]-data["x"][1]
    base_levels = pdict["base_levels"]
    bl_acc = Dict{Int,Vector{Float32}}()

    for bl in base_levels
    bl_acc[bl]=vec(sum(max.(bl.-all_s, 0),dims=1)) .* dx #m²
    end
    return bl_acc
end

function plot_acc_space(acc_space,pdict)
    base_levels = pdict["base_levels"]
    times = pdict["times"] #myr
    plots_param = pdict["plots_param"]

    fig, axs = subplots(2,1,sharex=true,figsize=(20*cm,20*cm))

    for i in eachindex(base_levels)
        bl = base_levels[i]
        s = acc_space[bl]
        dSdt = (s[2:end]-s[1:end-1]) ./ ((times[2:end]-times[1:end-1]) .* 1e6)
        
        axs[1].plot(times,s./1e6,color=plots_param["acc_colors"][i], label=bl)
        axs[2].plot(times[2:end],dSdt,color=plots_param["acc_colors"][i])
    end

    axs[1].legend()
    axs[1].set_ylabel("S (km²)")
    axs[2].set_ylabel("dS/dt (m²/y)")
    axs[2].set_xlabel("t (Myr)")
    axs[1].grid(true)
    axs[2].grid(true)
    axs[2].set_xlim(times[1],times[end])
    
    fig.tight_layout()
    fig.savefig("acc_dsdt.png",dpi=300)
    close(fig)
end

variable::String = "surface" #ARGS[1] # temperature, viscosity, strain_rate, stress, velocity, etc
data::NCDataset = NCDataset("$(variable).nc")
strain::NCDataset = NCDataset("strain.nc")
param = read_param("param.txt")
steps, times = read_times()
base_levels = Int[0, -500, -1000, -2000, -2500, -3000]
colors = ["#00008B", "#008080", "#1E90FF","#4682B4","#008000","#556B2F"]

Lx::Float32 = data["x"][end]
Nx::Int = length(data["x"][:])
Lz::Float32 = 0.0
Nz::Int = 0

if variable!="surface"
Lz::Float32 = data["z"][end]
Nz::Int = length(data["z"][:])
end

plots_param = Dict{String,Any}(
    "ylim" => (minimum(vec(data[variable][:,:]))-250, maximum(vec(data[variable][:,:]))+250),
    "grid" => true,
    "acc_colors" => colors,
)

global pdict = Dict{String,Any}(
    "Lx" => Lx,
    "Lz" => Lz,
    "Nx" => Nx,
    "Nz" => Nz,
    "dx" => Lx/(Nx-1),
    "dz" => Lz/(Nz-1),
    "h_air" => 40e3, #m
    "param" => param,
    "variable" => variable,
    "steps" => steps,
    "times" => times,
    "plots_param" => plots_param,
    "base_levels" => base_levels
)

#acc_space = Vector{Float32}(undef,length(steps))
acc = acc_space(data,pdict)
plot_acc_space(acc,pdict)

for i in eachindex(steps)
    step=steps[i]
    time=times[i]
    @printf("%01d-Processing step %d at time %.2f Myr\n",i, step, time)
    plot_surface(data,i,pdict,strain=false)
end

