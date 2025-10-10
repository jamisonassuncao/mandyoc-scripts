#!/usr/bin/env bash

#SBATCH --job-name=netcdf_convert    # Nome do job
#SBATCH --output=slurm-%A_%a.log     # Arquivo de saída para cada tarefa do array. %A=ID do Job, %a=ID da Tarefa
#SBATCH --mail-user=jbueno@usp.br
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --ntasks=1                   # Apenas uma tarefa principal por job do array
#SBATCH --cpus-per-task=16           # Solicita 16 CPUs para esta única tarefa
#SBATCH --time=12:00:00              # Tempo máximo de execução por tarefa

# --- Configuração ---
# Defina os cenários como um array bash.
# Adicione todos os seus cenários aqui, separados por espaço.
cenarios=(
   "teste09-stON"
   "teste09"
)

# O SLURM definirá a variável de ambiente $SLURM_ARRAY_TASK_ID com o índice do job atual.
# O índice do array em bash começa em 0.
current_scenario=${cenarios[$SLURM_ARRAY_TASK_ID]}
scripts_dir="/home/jbueno/scripts"
cenario_dir="/home/jbueno/cenarios_sed/$current_scenario"

# --- Execução ---
echo "========================================================"
echo "Iniciando processamento para o cenário: $current_scenario"
echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Rodando em: $(hostname)"
echo "Diretório do cenário: $cenario_dir"
echo "========================================================"

# Garante que o diretório do cenário existe antes de prosseguir
if [ ! -d "$cenario_dir" ]; then
    echo "Erro: O diretório do cenário '$cenario_dir' não foi encontrado."
    exit 1
fi

# Executa os scripts Julia usando o número de CPUs alocado por tarefa.
# A variável $SLURM_CPUS_PER_TASK é definida automaticamente pelo SLURM.
echo "Executando script convertNETCDF.jl..."
julia -t $SLURM_CPUS_PER_TASK "$scripts_dir/convertNETCDF.jl" "$cenario_dir"

echo "Executando script LithoNETCDF.jl..."
julia -t $SLURM_CPUS_PER_TASK "$scripts_dir/LithoNETCDF.jl" "$cenario_dir"

echo "Conversão para NETCDF concluída para: $current_scenario."
echo "Compactando e removendo diretórios de dados brutos..."

# Remove o diretório de litologia de forma mais explícita
# A sintaxe {lithos} funciona em bash, mas "$cenario_dir/lithos" é mais clara.
rm -rf "$cenario_dir/lithos"

# Lista de diretórios a serem compactados
data_dirs="heat viscosity velocity density pressure temperature surface strain_rate strain time"

# Compacta e remove os diretórios listados
tar -C "$cenario_dir" -czf "$cenario_dir/raw_data.tar.gz" --remove-files ${data_dirs}

echo "Processamento do cenário $current_scenario finalizado."
echo "========================================================"
