using NCDatasets
using Glob
using CSV
using Printf
using StatsBase
using DataFrames
using Base.Threads

# --- Funções Auxiliares (adaptadas do seu script) ---

function read_param(fpath::String="param.txt")::Dict{String,String}
    param_dict = Dict{String,String}()
    open(fpath, "r") do file
        for line in eachline(file)
            line=strip(line)
            if isempty(line) || startswith(line, "#")
                continue
            end
            line = split(line, "#")[1]
            line = replace(line, " " => "")
            key_value = split(line, "=")
            if length(key_value) == 2
                param_dict[lowercase(key_value[1])] = key_value[2]
            end
        end
    end
    return param_dict
end

function read_data(var::String, step::Int, nxnz::Tuple; veloc::Bool=false, surface::Bool=false)
    
    file = "$(var)/$(var)_$(step).txt"
    skipto::Int = 0
    if surface skipto=0 else skipto=3 end
    C = CSV.File(file, header=false, comment="P", skipto=skipto,types=Float32)|>CSV.Tables.matrix
    
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

function get_all_steps()::Vector{Int}
    pattern = joinpath("time", "time_*.txt")
    files = glob(pattern)
    
    times = [parse(Int, split(splitext(basename(f))[1], '_')[end]) for f in files]
    return sort(times)
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

function convert_to_nc(variable::String,pdict::Dict{String,Any})
    nc_fname = "$variable.nc"
    Nx = pdict["Nx"]
    Nz = pdict["Nz"]
    Lx = pdict["Lx"]
    Lz = pdict["Lz"]
    x_coords = pdict["x_coords"]
    z_coords = pdict["z_coords"]
    steps = pdict["steps"]
    num_steps = pdict["num_steps"]
    units = pdict["units"]
    h_air = pdict["air"]
    times = pdict["times"]
    dfllevel::Int8 = 5 #compression level 1-9


    if variable == "velocity" veloc=true else veloc=false end
    if variable == "surface" surface=true else surface=false end

    if surface
        sx_sample,_ = read_data("surface",0,(Nx,Nz),veloc=false, surface=true)
        surface_nx = size(sx_sample)[1]
        Nz = 0
        surface_x_coords = range(0.0f0, Lx, length=surface_nx)
        #println("Surface nx: $surface_nx")
        #println("Lx: $Lx")
        #println(surface_x_coords)
        
    end

    Dataset(nc_fname,"c") do ds #criar o arquivo nc

        defDim(ds,"time",num_steps)
        defVar(ds,"time",Float32.(times),("time",),attrib=Dict("units"=>units["time"],"long_name"=>"Time [Myr]","axis"=>"T"),
                                                                deflatelevel=dfllevel, shuffle=true)
        
        if veloc
            defDim(ds,"x",Nx)
            defDim(ds,"z",Nz)
            defVar(ds,"x",x_coords,("x",),attrib=Dict("units"=>"m","long_name"=>"x [m]"),
                                                                deflatelevel=dfllevel, shuffle=true)
            defVar(ds,"z",z_coords,("z",),attrib=Dict("units"=>"m","long_name"=>"z [m]"),
                                                                deflatelevel=dfllevel, shuffle=true)
            defVar(ds,"vx",Float32,("x","z","time"),attrib=Dict("units"=>units[variable],
                                                                "long_name"=>"vx"),
                                                                deflatelevel=dfllevel, shuffle=true)
            defVar(ds,"vy",Float32,("x","z","time"),attrib=Dict("units"=>units[variable],
                                                                "long_name"=>"vy",),
                                                               deflatelevel=dfllevel, shuffle=true)
        elseif surface
            defDim(ds,"x",surface_nx)
            defVar(ds,"x",surface_x_coords,("x",),attrib=Dict("units"=>"m","long_name"=>"x [m]",
                                                                "deflatelevel"=>dfllevel))
            defVar(ds,variable,Float32,("x","time"),attrib=Dict("units"=>units[variable],"long_name"=>variable),
                                                                )
        else
            defDim(ds,"x",Nx)
            defDim(ds,"z",Nz)
            defVar(ds,"x",x_coords,("x",),attrib=Dict("units"=>"m","long_name"=>"x [m]",
                                                                "deflatelevel"=>dfllevel))
            defVar(ds,"z",z_coords,("z",),attrib=Dict("units"=>"m","long_name"=>"z [m]",
                                                                "deflatelevel"=>dfllevel))
            defVar(ds,variable,Float32,("x","z","time"),attrib=Dict("units"=>units[variable],"long_name"=>variable),
                                                                deflatelevel=dfllevel, shuffle=true)
        end
	nc_lock = ReentrantLock()
        @threads for i in eachindex(steps)
            step = steps[i]
            @printf("Processing step: %d (%d of %d)\r", step, i, num_steps)
            #fpath = "$(variable)_$step"
            data = read_data(variable,step,(Nx,Nz),veloc=veloc, surface=surface)
            if data !== nothing
            lock(nc_lock) do
	    if veloc
                dens = read_data("density",step,(Nx,Nz),veloc=false, surface=false)
                vx,vy = data
                vx[dens.<1200] .= 0
                vy[dens.<1200] .= 0
                ds["vx"][:,:,i] = vx'
                ds["vy"][:,:,i] = vy'
            elseif surface

                sx,sy = data
                sy .= sy .+ h_air
                fix_topo = mean(sy[(sx.>100e3).&(sx.< 250e3)]) #m
                sy .= sy .- fix_topo

                ds[variable][:,i] = sy
            else
                dens = read_data("density",step,(Nx,Nz),veloc=false, surface=false)
                data[dens.<1200] .= 0
                ds[variable][:,:,i] = data'
            end
	    end
            else
                @warn "No data found for $fpath at step $step"
            end
    
        end
        println("\nSaved to $nc_fname")
    end
end

# --- Lógica Principal do Script ---

# Verifique se o diretório foi passado como argumento
data_dir = ARGS[end]
cd(data_dir)

# 1. Variáveis para passar pra binario
vars = ["density", "viscosity", "pressure", "strain","strain_rate","temperature","velocity","surface","heat"]

# 2. Parametros
params = read_param("param.txt")
Nx = parse(Int, params["nx"])
Nz = parse(Int, params["nz"])
Lx = parse(Float32, params["lx"])
Lz = parse(Float32, params["lz"])

# 3. Encontrar todos os steps
steps = get_all_steps()
num_steps = length(steps)
times = Float32[read_time(step) for step in steps]
println("Encontrados $num_steps passos de tempo, de $(steps[1]) a $(steps[end]).")
CSV.write("times.csv", DataFrame(step=steps, time_myr=times))

units=Dict{String,String}(
    "density"=>"kg/m³",
    "viscosity"=>"Pa.s",
    "pressure"=>"Pa",
    "strain"=>"dimensionless",
    "strain_rate"=>"1/s",
    "temperature"=>"°C",
    "velocity"=>"m/s",
    "surface"=>"m",
    "time"=>"Myr",
    "heat"=>"--"
    
)

# 4. Definir coordenadas
x_coords = range(0.0f0, Lx, length=Nx)
z_coords = range(0.0f0, Lz, length=Nz)

pdict = Dict{String,Any}("Nx"=>Nx, "Nz"=>Nz, "Lx"=>Lx, "Lz"=>Lz,
    "steps"=>steps, "num_steps"=>num_steps, "x_coords"=>x_coords, "z_coords"=>z_coords,
    "units"=>units, "air"=>40.0e3, "times"=>times
)

# 5. NetCDF para cada variável
for var in vars
    println("Convertendo variável: $var")
    convert_to_nc(var,pdict)
end

println("\nConversão concluída!")
