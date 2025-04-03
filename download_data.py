import requests

url = "https://ubcca-my.sharepoint.com/:u:/r/personal/obashp19_student_ubc_ca/Documents/structured_data/PDEBench/ns_incom_dt1_128.h5?csf=1&web=1&e=9TOouF"
output_file = "ns_incom_dt1_128.h5"

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Download complete:", output_file)
else:
    print("Failed to download. Status code:", response.status_code)