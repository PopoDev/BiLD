import xml.etree.ElementTree as ET
import json
import paramiko
import fire

from io import IOBase, StringIO
from typing import Dict, List
from multiprocessing import Pool
from functools import partial

def parseUtilDetails(xml_io: IOBase, verbose: bool = False):
    tree = ET.parse(xml_io)
    root = tree.getroot()
    results = dict()

    for gpu_item in root.findall("gpu"):
        curr_results = dict()
        gpu_id = gpu_item.get("id")

        memory_item = gpu_item.find("fb_memory_usage")

        if verbose:
            curr_results["total_mem"] = memory_item.find("total").text
            curr_results["reserved_mem"] = memory_item.find("reserved").text
            curr_results["used_mem"] = memory_item.find("used").text
            curr_results["free_mem"] = memory_item.find("free").text

        utilization_item = gpu_item.find("utilization")

        curr_results["GPU Util"] = utilization_item.find("gpu_util").text
        curr_results["GPU Mem Util"] = utilization_item.find("memory_util").text

        results[f"GPU {gpu_id}"] = curr_results

    return results

def getUtilDetails(
    remote_host: str,
    ssh_username: str,
    ssh_password: str,
    verbose: bool = False
) -> Dict[str, str]:
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh_client.connect(
            remote_host,
            username = ssh_username,
            password = ssh_password
        )

        stdin, stdout, stderr = ssh_client.exec_command("nvidia-smi -q -x")
        usage_output = output = stdout.read().decode('utf-8')

        return parseUtilDetails(StringIO(usage_output), verbose = verbose)
    finally:
        ssh_client.close()

def dumpAllUsageDetails(
    ssh_username: str,
    ssh_password: str,
    remote_hosts: List[str] = ["nlpg00", "nlpg01", "nlpg02", "nlpg03"],
    verbose: bool = False,
    num_sub_processes: int = 4
) -> None:
    NUM_PROCESSES = 4
    output_dict = dict()

    with Pool(num_sub_processes) as p:
        output_list = p.map(
            partial(
                getUtilDetails,
                ssh_username = ssh_username,
                ssh_password = ssh_password,
                verbose = verbose
            ),
            remote_hosts
        )

    output_dict = dict()
    for curr_host, curr_output in zip(remote_hosts, output_list):
        output_dict[curr_host] = curr_output

    print(json.dumps(output_dict, indent = 4))

if __name__ == "__main__":
    fire.Fire(dumpAllUsageDetails)
