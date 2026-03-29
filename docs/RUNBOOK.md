# Runbook: Personal Likeness LoRA Training

Complete step-by-step guide from zero to a trained `.safetensors` file.

---

## 1. Prep (Local)

### Curate Your Photos

Collect 20-30 personal photos. Quality matters more than quantity.

**Good photos:**
- Variety of angles: front, 3/4, profile, slightly above/below
- Variety of lighting: natural daylight, indoor, studio, golden hour
- Variety of expressions: neutral, smiling, serious, laughing
- Variety of backgrounds: plain walls, outdoors, office, etc.
- Clear, sharp focus on your face
- Minimum resolution: 512x512 (ideally 1024x1024 or higher)
- Mix of close-up headshots and upper-body shots

**Avoid:**
- Sunglasses or hats in more than 2-3 photos (model won't learn what's hidden)
- Heavy filters or extreme color grading
- Group photos where your face is small
- Very similar/duplicate photos (wastes training capacity)
- Blurry or low-resolution images
- Photos where your face is partially occluded

Place your selected photos in `datasets/raw_photos/` within your local clone of this repo (the directory is gitignored).

---

## 2. Launch EC2

### Create VPC and Security Group (one-time)

```bash
# ── Create a VPC ──
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --region us-east-1 \
    --query 'Vpc.VpcId' --output text)
aws ec2 create-tags --resources "$VPC_ID" --tags Key=Name,Value=lora-training --region us-east-1
echo "VPC: $VPC_ID"

# ── Create a public subnet ──
SUBNET_ID=$(aws ec2 create-subnet \
    --vpc-id "$VPC_ID" \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-east-1a \
    --region us-east-1 \
    --query 'Subnet.SubnetId' --output text)
aws ec2 create-tags --resources "$SUBNET_ID" --tags Key=Name,Value=lora-training-public --region us-east-1
echo "Subnet: $SUBNET_ID"

# ── Create and attach an internet gateway (needed for SSH + model downloads) ──
IGW_ID=$(aws ec2 create-internet-gateway \
    --region us-east-1 \
    --query 'InternetGateway.InternetGatewayId' --output text)
aws ec2 attach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID" --region us-east-1
aws ec2 create-tags --resources "$IGW_ID" --tags Key=Name,Value=lora-training --region us-east-1
echo "Internet Gateway: $IGW_ID"

# ── Add a default route to the internet gateway ──
RTB_ID=$(aws ec2 describe-route-tables \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --region us-east-1 \
    --query 'RouteTables[0].RouteTableId' --output text)
aws ec2 create-route --route-table-id "$RTB_ID" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW_ID" --region us-east-1
echo "Route table: $RTB_ID"

# ── Create security group ──
SG_ID=$(aws ec2 create-security-group \
    --group-name lora-training-sg \
    --description "SSH access for LoRA training" \
    --vpc-id "$VPC_ID" \
    --region us-east-1 \
    --query 'GroupId' --output text)
echo "Security Group: $SG_ID"

# ── Allow SSH from your current IP ──
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 22 \
    --cidr "$(curl -s https://checkip.amazonaws.com)/32" \
    --region us-east-1
```

> **Tip:** Save the `VPC_ID`, `SUBNET_ID`, and `SG_ID` values — you'll need them for the launch command and cleanup.

### Find the Latest DLAMI AMI

```bash
# Find the latest PyTorch DLAMI for us-east-1
aws ec2 describe-images \
    --region us-east-1 \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning AMI GPU PyTorch*" \
        "Name=state,Values=available" \
        "Name=architecture,Values=x86_64" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].{AMI:ImageId, Name:Name}' \
    --output table
```

Note the AMI ID from the output (e.g., `ami-0abcdef1234567890`).

### Launch the Instance

```bash
aws ec2 run-instances \
    --region us-east-1 \
    --image-id <AMI_ID_FROM_ABOVE> \
    --instance-type g5.xlarge \
    --key-name your-key-pair \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --associate-public-ip-address \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=lora-training}]' \
    --query 'Instances[0].InstanceId' \
    --output text
```

> **Note:** 200 GB storage is needed for model weights (~24 GB for Flux Kontext) plus datasets and outputs.

### Get the Public IP

```bash
# Wait for the instance to be running
aws ec2 wait instance-running --instance-ids <INSTANCE_ID> --region us-east-1

# Get the public IP
aws ec2 describe-instances \
    --instance-ids <INSTANCE_ID> \
    --region us-east-1 \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text
```

---

## 3. Bootstrap

SSH into the instance and run the bootstrap script:

```bash
ssh -i ~/.ssh/your-key-pair.pem ubuntu@<EC2_PUBLIC_IP>

# Clone the repo and run setup
git clone https://github.com/The-CarL/likeness-lab.git ~/likeness-lab
cd ~/likeness-lab
bash scripts/bootstrap_ec2.sh
```

This will:
- Verify GPU availability
- Install ai-toolkit and dependencies
- Log into HuggingFace (have your token ready)
- Pre-download Flux Kontext Dev model weights (~24 GB, takes 15-30 min)

---

## 4. Upload Dataset

From your **local machine**, upload your photos:

```bash
scp -i ~/.ssh/your-key-pair.pem -r datasets/raw_photos/* \
    ubuntu@<EC2_PUBLIC_IP>:~/likeness-lab/datasets/raw_photos/
```

Verify on EC2:

```bash
ls ~/likeness-lab/datasets/raw_photos/ | wc -l
# Should show 20-30 files
```

---

## 5. Prepare Kontext Pairs

Generate the before/after paired dataset:

```bash
cd ~/likeness-lab

# Preview what will be created
python scripts/prepare_kontext_pairs.py --dry-run

# Generate pairs (uses face blur by default — fast, no extra VRAM)
python scripts/prepare_kontext_pairs.py

# Or use diffusion inpainting for higher quality (needs GPU, slower)
# python scripts/prepare_kontext_pairs.py --method inpaint
```

**Review the output:**
```bash
# Check that control images look right (faces should be blurred/replaced)
ls datasets/kontext_pairs/target/ | head
ls datasets/kontext_pairs/control/ | head

# Verify file counts match
echo "Target: $(ls datasets/kontext_pairs/target/*.jpg 2>/dev/null datasets/kontext_pairs/target/*.png 2>/dev/null | wc -l)"
echo "Control: $(ls datasets/kontext_pairs/control/*.jpg 2>/dev/null datasets/kontext_pairs/control/*.png 2>/dev/null | wc -l)"
```

You can SCP a few control images back to your Mac to visually inspect them:
```bash
# From local machine
scp -i ~/.ssh/your-key-pair.pem ubuntu@<EC2_IP>:~/likeness-lab/datasets/kontext_pairs/control/001.jpg /tmp/
open /tmp/001.jpg
```

---

## 6. Caption

Validate and fix captions for the Kontext dataset:

```bash
cd ~/likeness-lab

# Preview
python scripts/caption_dataset.py --mode kontext --dry-run

# Run
python scripts/caption_dataset.py --mode kontext
```

If using the standard (non-Kontext) config instead:
```bash
python scripts/caption_dataset.py --mode standard --dataset datasets/standard
```

---

## 7. Train

```bash
cd ~/likeness-lab

# Start training with the Kontext config (default)
bash scripts/launch_training.sh

# Or explicitly specify a config
bash scripts/launch_training.sh --config kontext_personal_likeness.yaml

# For standard Flux Dev LoRA
bash scripts/launch_training.sh --config flux_dev_standard.yaml
```

Training will:
- Run for 1500 steps (~1-3 hours on g5.xlarge)
- Save checkpoints every 500 steps to `outputs/`
- Generate sample images every 500 steps
- Log everything to `logs/training_<timestamp>.log`

You can safely disconnect SSH — training continues. Reconnect and check with:
```bash
tail -f ~/likeness-lab/logs/training_*.log
```

---

## 8. Monitor Training

### Check Progress

```bash
# Watch the log in real-time
tail -f ~/likeness-lab/logs/training_*.log

# Check GPU utilization
nvidia-smi

# Watch GPU continuously
watch -n 2 nvidia-smi
```

### What to Look For

- **Loss should decrease** over the first few hundred steps, then plateau. A typical final loss is 0.05-0.15.
- **Sample images** appear in `outputs/` — check them periodically to see if the likeness is converging.
- **GPU memory** should be ~20-22 GB used out of 24 GB. If you see OOM errors, reduce resolution in the config.
- **Steps/second** — expect ~1-2 steps/sec on A10G. Much slower means something is wrong.

### Check Samples

```bash
ls -la outputs/*/samples/
# SCP samples to your Mac for inspection
scp -i ~/.ssh/your-key-pair.pem "ubuntu@<EC2_IP>:~/likeness-lab/outputs/*/samples/*.png" /tmp/samples/
```

---

## 9. Validate on EC2 (ComfyUI)

After training completes, set up ComfyUI for quick validation on the EC2 instance:

```bash
# Install and start ComfyUI
bash scripts/setup_comfyui.sh
```

From your **local machine**, create an SSH tunnel:

```bash
ssh -L 8188:localhost:8188 -i ~/.ssh/your-key-pair.pem ubuntu@<EC2_PUBLIC_IP>
```

Open [http://localhost:8188](http://localhost:8188) in your browser. Load a Flux Kontext workflow and test your LoRA.

**Test prompts to try:**
- "Replace this person with CARLOP" (with a control image of someone else)
- "Make this person look like CARLOP, professional headshot"
- "Turn this into a photo of CARLOP at the beach"

If ComfyUI is already installed and you just need to restart it:
```bash
bash scripts/setup_comfyui.sh --start-only
```

---

## 10. Download Weights

From your **local machine**, pull the trained weights:

```bash
cd ~/likeness-lab   # or wherever you cloned this repo locally

# Download all checkpoints
bash scripts/download_weights.sh <EC2_PUBLIC_IP>

# Or just the latest
bash scripts/download_weights.sh <EC2_PUBLIC_IP> --latest

# With a specific SSH key
bash scripts/download_weights.sh <EC2_PUBLIC_IP> --key ~/.ssh/your-key-pair.pem
```

This copies the `.safetensors` to both `~/lora-backups/` and `~/ComfyUI/models/loras/`.

---

## 11. Validate Locally

Open your local ComfyUI and test the LoRA:

1. Start ComfyUI: `cd ~/ComfyUI && python main.py`
2. Open [http://localhost:8188](http://localhost:8188)
3. Build or load a Flux Kontext workflow
4. Add a LoRA Loader node pointing to your new `.safetensors`
5. Test with various prompts and control images

**Key things to validate:**
- Likeness accuracy at different angles
- Style consistency across different prompts
- No artifacts or distortions
- Works at different LoRA strengths (try 0.6, 0.8, 1.0)

---

## 12. Iterate

If results aren't satisfactory, adjust and retrain.

### Underfitting (face looks generic, not like you)

- **Increase steps**: 1500 → 2000 → 2500
- **Increase learning rate**: 1e-4 → 2e-4
- **Increase rank**: 16 → 32
- **Improve dataset**: add more varied photos, ensure faces are well-lit and sharp

### Overfitting (copies training images exactly, can't generalize)

- **Decrease steps**: 1500 → 1000 → 800
- **Decrease learning rate**: 1e-4 → 5e-5
- **Increase caption dropout**: 0.05 → 0.1
- **Improve dataset diversity**: more backgrounds, outfits, lighting conditions

### Artifacts or Distortions

- **Lower learning rate**: 1e-4 → 5e-5
- **Use an earlier checkpoint** (check the intermediate saves at 500, 1000 steps)
- **Reduce rank**: 16 → 8 (less capacity = less overfitting)

### Poor Control Image Quality (Kontext-specific)

- Use `--method inpaint` instead of blur for prepare_kontext_pairs.py
- Manually review and fix control images that look bad
- Ensure control/target filenames match exactly

---

## 13. Cleanup

**Terminate the EC2 instance** when you're done to stop charges:

```bash
aws ec2 terminate-instances \
    --instance-ids <INSTANCE_ID> \
    --region us-east-1
```

**Verify no lingering resources:**

```bash
# Check for running instances
aws ec2 describe-instances \
    --region us-east-1 \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime]' \
    --output table

# Check for orphaned EBS volumes (should be auto-deleted with instance)
aws ec2 describe-volumes \
    --region us-east-1 \
    --filters "Name=status,Values=available" \
    --query 'Volumes[*].[VolumeId,Size,CreateTime]' \
    --output table
```

If you want to tear down the entire VPC stack (reverse order):
```bash
# Delete security group
aws ec2 delete-security-group --group-id "$SG_ID" --region us-east-1

# Detach and delete internet gateway
aws ec2 detach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID" --region us-east-1
aws ec2 delete-internet-gateway --internet-gateway-id "$IGW_ID" --region us-east-1

# Delete subnet
aws ec2 delete-subnet --subnet-id "$SUBNET_ID" --region us-east-1

# Delete VPC
aws ec2 delete-vpc --vpc-id "$VPC_ID" --region us-east-1
```

---

## Cost Tracking

| Resource | Cost | Duration |
|----------|------|----------|
| g5.xlarge (on-demand) | $1.006/hr | Setup: ~30 min, Training: 1-3 hrs, Validation: ~30 min |
| EBS storage (200 GB gp3) | $0.08/GB/month | Pro-rated to hours used |
| Data transfer | ~$0 | Minimal egress for downloading weights |

**Typical total cost: $2-5 per training run.**

To monitor costs in real-time:
```bash
# Rough estimate based on instance uptime
aws ec2 describe-instances \
    --instance-ids <INSTANCE_ID> \
    --region us-east-1 \
    --query 'Reservations[0].Instances[0].LaunchTime' \
    --output text
```

---

## Troubleshooting

### OOM (Out of Memory) Errors

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix:** Reduce resolution in the config. For Kontext, use `[512, 768]` (not 1024). Ensure `quantize: true` and `gradient_checkpointing: true` are set.

### "Access denied" Downloading Model

```
HTTP Error 403: Forbidden
```

**Fix:** Accept the model license at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev and verify your HuggingFace token with `huggingface-cli whoami`.

### Training Loss Not Decreasing

- Check that captions contain the trigger word (`CARLOP`)
- Verify dataset isn't corrupted: `python -c "from PIL import Image; Image.open('datasets/kontext_pairs/target/001.jpg').verify()"`
- Try a higher learning rate (2e-4)

### ComfyUI Won't Start

```bash
# Check if port is already in use
lsof -i :8188

# Kill existing process and restart
kill $(lsof -ti :8188) 2>/dev/null
bash scripts/setup_comfyui.sh --start-only
```

### SSH Tunnel Not Working

```bash
# Make sure you're using -L (local forward), not -R
ssh -L 8188:localhost:8188 -i ~/.ssh/your-key.pem ubuntu@<ip>

# Verify ComfyUI is running on EC2
curl -s http://localhost:8188 | head -5
```

### "No face detected" in prepare_kontext_pairs.py

- Ensure photos are well-lit with clearly visible faces
- Try photos with faces facing more toward the camera
- The script falls back to full-image blur if no face is found — these pairs still work but are lower quality

### Training Seems Stuck / Very Slow

```bash
# Check GPU utilization
nvidia-smi
# GPU-Util should be 90-100%. If 0%, training may have crashed.

# Check the log for errors
tail -50 ~/likeness-lab/logs/training_*.log
```
