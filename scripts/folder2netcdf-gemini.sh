#!/usr/bin/env bash

#SBATCH --job-name=netcdf_convert  # Nome do job
#SBATCH --output=slurm-%A_%a.log   # Arquivo de saída para cada tarefa do arra %A=ID do Job, %a=ID da Tarefa
#SBATCH --mail-user=jbueno@usp.br
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1                 # Apenas uma tarefa principal por job do array
#SBATCH --cpus-per-task=20         # Solicita 16 CPUs para esta única tarefa
#SBATCH --time=12:00:00            # Tempo por tarefa ajuste conforme necessário, 12h pode ser mais seguo

# --- Configuração ---
# Defina os cenários como um array bash
cenarios=(
   "again-teste03"
    "again2-teste06"
    "teste04-2"
    "teste08"
    "teste09-spOFF"
    "teste10"
    "again-teste05"
    "teste02"
    "teste07"
   )

# O SLURM definirá a variável $SLURM_ARRAY_TASK_ID com o índice do job atual.
# Para um array de 12 itens, você submeteria com: sbatch --array=0-11 folter2netcdf.sh
current_scenario=$1
scripts_dir="/home/jbueno/scripts"
cenario_dir=$1


# --- Execução ---
echo "Iniciando processamento para o cenário: $current_scenario (Array Task ID: $SLURM_ARRAY_TASK_ID)"

# Executa os scripts Julia usando o número de CPUs alocado por tarefa
# A variável $SLURM_CPUS_PER_TASK é definida automaticamente pelo SLURM
julia -t $SLURM_CPUS_PER_TASK "$scripts_dir/convertNETCDF.jl" "$cenario_dir"
julia -t $SLURM_CPUS_PER_TASK "$scripts_dir/LithoNETCDF.jl" "$cenario_dir"

echo "Conversão para NETCDF concluída: $current_scenario. Zipando e apagando diretórios..."

# Remove os diretórios de dados brutos em uma única linha (corrigido "lithos")
rm -rf "$cenario_dir"/lithos
tar -C "$cenario_dir" -czf "$cenario_dir/raw_data.tar.gz" --remove-files \
    heat viscosity velocity density pressure temperature surface strain_rate strain time

echo "Processamento do cenário $current_scenario finalizado."

# O "echo finished!" não é necessário aqui, pois o log mostrará quando cada tarefa terminar.
# O e-mail de "END" será enviado quando todo o array de jobs for concluído.

#sbatch --array=0-11 /media/jobueno/STOV/scripts/folter2netcdf.sh
#Isso submeterá 12 jobs. O SLURM executará quantos puder em paralelo, 
#de acordo com os recursos disponíveis no cluster. 
#Cada um executará o script para um cenário diferente, tornando o uso dos recursos do HPC muito mais eficiente.
