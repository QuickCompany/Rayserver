# Rayserver
#Rayserve exmaple documents
https://docs.ray.io/en/latest/serve/getting_started.html

ray start --head --dashboard-host=0.0.0.0

serve run serving_model:translator_app

sudo nano /etc/systemd/system/ray-serve.service


sudo systemctl daemon-reload
sudo systemctl enable ray-serve.service
sudo systemctl start ray-serve.service

systemctl status ray-serve

journalctl -u ray-serve.service


rsync -avz -e "ssh -i gpu-key.pem" ./parsing_models ubuntu@ec2-3-110-90-123.ap-south-1.compute.amazonaws.com:/home/ubuntu
