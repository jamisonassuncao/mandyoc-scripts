global dir = ARGS[end]
cd(dir) # Change to the specified directory

using Distributed
addprocs(Sys.CPU_THREADS - 1)

@everywhere begin
using PyPlot #v2.11.6
using CSV 
using DataFrames
using StatsBase
using Printf
using NCDatasets #v0.10.4

const cm = 1/2.54 #centimeter in inches
const cr = 255.0 #RGB color scale factor

PyPlot.ioff() #desliga processo interativo
end

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

function read_times()::Tuple{Vector{Int}, Vector{Float32}}
    columns_types = Dict{String,DataType}("step" => Int, "time_myr" => Float32)
    ts = CSV.File("times.csv",header=true,types=columns_types)|>CSV.Tables.matrix
    return (ts[:,1], ts[:,2])
end

@everywhere begin
function plot_step(stepindex::Int,pdict::Dict)::Nothing #Nothing: Datatype with a single value 'nothing'
    litho = NCDataset("lithology.nc")
    strain = NCDataset("strain.nc")
    temperature = NCDataset("temperature.nc")

    extent = pdict["extent"]
    time = pdict["times"][stepindex]
    step = pdict["steps"][stepindex]
    plots_param = pdict["plots_param"]

    fig,axs = subplots(1,1,(figsize=(24*cm,8*cm)))

    #xi = data["x"][:]./1e3  #km
    #zi = data["z"][:]./-1e3 ,+ h_air/1e3 #km


    axs.imshow(reverse!(litho["lithology"][:,:,stepindex]', dims=1),cmap=plots_param["litho_colors"],
        vmin=0,vmax=9,extent=extent,
        aspect="auto")
    
    axs.imshow(reverse!(strain["strain"][:,:,stepindex]', dims=1),cmap="Greys", alpha=0.2,
        vmin=-0.5, vmax=0.9, aspect="auto", extent=extent
        )
    
    axs.contour(temperature["temperature"][:,:,stepindex]',levels=pdict["temp_intervals"],
        colors="r", linewidths=0.75, extent=extent
        )
    
    axs.set_xlim(plots_param["xlim"])
    axs.set_ylim(plots_param["ylim"])
    axs.grid(plots_param["grid"])
    axs.set_xlabel("X [km]")
    axs.set_ylabel("Z [m]")
    axs.set_title("$(round(time,digits=3)) Myr - step_$(step)")
    fig.tight_layout()
    fname = Printf.@sprintf("litho_%05d.png",step)
    fig.savefig(fname,dpi=300)
    close(fig)

    close(litho)
    close(strain)
    close(temperature)
    return nothing
end
end

function main()
    litho_main = NCDataset("lithology.nc")
    temperature = NCDataset("temperature.nc")
    steps, times = read_times()
    

    Lx::Float32 = temperature["x"][end]
    Lz::Float32 = temperature["z"][end]
    Nx::Int = length(temperature["x"][:])
    Nz::Int = length(temperature["z"][:])
    Nxlitho::Int = length(litho_main["x"][:])
    Nzlitho::Int = length(litho_main["z"][:])
    temp_intervals = Int32[600, 800, 1000, 1200, 1350]
    h_air::Float32 = 40e3 #m

    plot_extent = Float32[temperature["x"][1]/1e3, temperature["x"][end]/1e3, 
        -1 .*(temperature["z"][end]/1e3 - h_air/1e3),-1 .*(temperature["z"][1]/1e3 - h_air/1e3)] #km

color_air  = Float16[1,1,1]
color_sed1 = Float16[241/cr, 184/cr, 68/cr]
color_sed2 = Float16[128/cr, 95/cr, 40/cr]
color_sed3 = Float16[167/cr, 245/cr, 66/cr]
color_salt = Float16[245/cr, 66/cr, 218/cr]
color_uc   = Float16[165/cr, 163/cr, 238/cr]
color_lc   = Float16[227/cr, 217/cr, 242/cr]
color_lit  = Float16[155/cr, 194/cr, 155/cr]
color_ast  = Float16[207/cr, 226/cr, 205/cr]

    colors = Vector{Float16}[color_ast,color_lit,color_lc,color_uc,color_salt,color_sed1,color_sed2,color_sed3,color_air]
    
    plots_param = Dict{String,Any}(
        "ylim" => (-200, 20), #km
        "xlim" => (200, 1000), #km
        "grid" => false,
        "litho_colors" => PyPlot.matplotlib.colors.ListedColormap(colors,"my_cmap"),
        
    )

    global pdict = Dict{String,Any}(
        "Lx" => Lx,
        "Lz" => Lz,
        "Nx" => Nx,
        "Nz" => Nz,
        "Nxlitho" => Nxlitho,
        "Nzlitho" => Nzlitho,
        "h_air" => h_air, #m
        "steps" => steps,
        "times" => times,
        "plots_param" => plots_param,
        "temp_intervals" => temp_intervals,
        "extent" => plot_extent,
    )

    @everywhere global pdict = $pdict

    close(litho_main)
    close(temperature)

    println("Starting parallel processing of $(length(steps)) steps - $(Sys.CPU_THREADS - 1) workers")
    @sync @distributed for i in eachindex(steps)
        step=steps[i]
        @printf("%01d-Processing step %d at time %.2f Myr\n",i, step, times[i])
        plot_step(i,pdict)
        GC.gc() #garbage collector
    end
    
    rmprocs(workers()) #remove os workers
    println("All workers removed.")
    println("done")
end

main()