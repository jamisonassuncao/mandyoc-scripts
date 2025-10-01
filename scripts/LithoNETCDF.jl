using NCDatasets
using Glob
using CSV
using Printf
using DataFrames
using Base.Threads

# --- Funções Auxiliares (adaptadas de seus scripts) ---

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

function read_litho_file(fpath::String)
    # Read lithology file and return x, z, lith
    CLit = CSV.read(fpath, DataFrame, header=false, comment="P", delim=" ")
    return (CLit[1:end,1], CLit[1:end,2], CLit[1:end,3])
end

function replace_negatives_with_neighbors!(mat::Matrix)
    result = mat
    rows::Int16, cols::Int16 = size(mat)
    negative_indices = findall(result .< 0)

    for idx in negative_indices
        i::Int16, j::Int16 = idx[1], idx[2]
        found = false
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ni, nj = i + di, j + dj
            if (1 <= ni <= rows && 1 <= nj <= cols) && mat[ni, nj] >= 0
                result[i, j] = mat[ni, nj]
                found = true
                break
            end
        end
    end
    return result
end


#plotando de cabeça pra baixo
function convert_litho_to_nc(pdict::Dict{String,Any})
    nc_fname = "lithology.nc"
    Nx = pdict["Nx"]
    Nz = pdict["Nz"]
    Lx = pdict["Lx"]
    Lz = pdict["Lz"]
    steps = pdict["steps"]
    num_steps = pdict["num_steps"]
    times = pdict["times"]
    ncores = pdict["ncores"]
    litho_dict = pdict["litho_dict"]
    dfllevel::Int8 = 2 #compression level 1-9
    
    # A malha de litologia tem resolução maior
    Nxl = (Nx-1)*5 + 1
    Nzl = (Nz-1)*5 + 1
    
    

    x_coords_litho = range(0.0f0, Lx, length=Nxl)
    z_coords_litho = range(0.0f0, Lz, length=Nzl)
    
    Dataset(nc_fname, "c") do ds
        # --- Definir Dimensões e Variáveis ---
        defDim(ds, "time", num_steps)
        defDim(ds, "x", Nxl)
        defDim(ds, "z", Nzl)

        defVar(ds, "time", Float32.(times), ("time",), 
            attrib=Dict("units"=>"Myr", "long_name"=>"Time [Myr]","axis"=>"T"),
            deflatelevel=dfllevel, shuffle=true)
        defVar(ds, "x", x_coords_litho, ("x",), 
            attrib=Dict("units"=>"m", "long_name"=>"x [m]"),
            deflatelevel=dfllevel, shuffle=true)
        defVar(ds, "z", z_coords_litho, ("z",), 
            attrib=Dict("units"=>"m", "long_name"=>"z [m]"),
            deflatelevel=dfllevel, shuffle=true)
        
        defVar(ds, "lithology", Int8, ("x", "z", "time"), 
            attrib=Dict(
                "long_name"=>"Lithology",
                "comment"=>"Codes are mapped: 1=Mantle, 2=LC, 3=UC, 4=Sed1, 5=Sed2, 6=Salt, 7=Air"
            ),
            deflatelevel=dfllevel, shuffle=true)

        # --- Loop sobre os passos de tempo para preencher os dados ---
        nc_lock = ReentrantLock()
	@threads for i in eachindex(steps)
            step = steps[i]
            @printf("Processing lithology for step: %d (%d of %d)\r", step, i, num_steps)
            
            # --- DEBUG: Adicione um conjunto para rastrear códigos de litologia ---
            #unique_codes = Set{Int}()
            # ----------------------------------------------------------------

            # 1. Criar a malha de litologia para o passo atual
            litho_mesh = zeros(Int8, Nzl, Nxl) .- Int8(1)

            for core in 0:(ncores-1)
                fpath=joinpath("lithos","litho_$(step)_$core.txt")
                if isfile(fpath)
                    x_core, z_core, litho_core = read_litho_file(fpath)

                    # --- DEBUG: Colete os códigos únicos ---
                    #union!(unique_codes, litho_core)
                    # ------------------------------------

                    for k in eachindex(x_core)
                        x_idx = x_core[k] + 1
                        z_idx = z_core[k] + 1
                        litho_mesh[z_idx, x_idx] = Int8(litho_core[k])
                    end
                end
            end

            # 2. Processar a malha (preencher vazios e mapear valores)
            replace_negatives_with_neighbors!(litho_mesh)
            litho_mesh .= get.(Ref(litho_dict), litho_mesh, litho_mesh)
            # --- DEBUG: Imprima os códigos encontrados para este step ---
            #if i == 1 # Imprime apenas para o primeiro step para não poluir o log
            #    @printf("\nCódigos de litologia encontrados nos arquivos para o step %d: %s\n", step, unique_codes)
            #    @printf("Chaves no dicionário: %s\n", keys(litho_dict))
            #end
            # ---------------------------------------------------------
	    lock(nc_lock) do
            # 3. Escrever no arquivo NetCDF (lembre-se de transpor)
            ds["lithology"][:, :, i] = reverse(litho_mesh, dims=1)'
	    end
        end
        println("\nSaved to $nc_fname")
    end
end

data_dir = ARGS[1]
cd(data_dir)

params = read_param("param.txt")
Nx = parse(Int, params["nx"])
Nz = parse(Int, params["nz"])
Lx = parse(Float32, params["lx"])
Lz = parse(Float32, params["lz"])

steps::Vector{Int} = get_all_steps()
num_steps::Int = length(steps)
times = Float32[read_time(step) for step in steps]
#name::String = split(basename(glob(joinpath("lithos","litho_0_*.txt"))),'.')[1]
ncores::Int = size(glob(joinpath("lithos","litho_0_*.txt")))[1]
println("Encontrados $num_steps passos de tempo, de $(steps[1]) a $(steps[end]).")
println("Número de núcleos detectados: $ncores")

#mudar litologia com base nas camadas do interfaces
litho_dict = Dict{Int8,Int8}(
    2 => 1,  # Lithospheric mantle
    3 => 1,  # Lithospheric mantle
    4 => 2,  # Lower crust
    5 => 3,  # Upper crust
    6 => 4,  # Sed 1
    7 => 5,  # Sed 2
    8 => 6,  # Salt
    9 => 7,  # ?? - air
    10 => 8   # air
)

pdict = Dict{String,Any}(
    "Nx"=>Nx, "Nz"=>Nz, "Lx"=>Lx, "Lz"=>Lz,
    "steps"=>steps, "num_steps"=>num_steps, "times"=>times,
    "ncores"=>ncores, "litho_dict"=>litho_dict
)

println("Convertendo dados de litologia...")
convert_litho_to_nc(pdict)
println("\nConversão concluída!")
