---
layout: docs
title: AWS Compute Services
hide_title: true
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "server"
---

<div class="hero-section" style="background: linear-gradient(135deg, #ff9900 0%, #ffb84d 100%); color: white; padding: 3rem 2rem; margin: -2rem -3rem 2rem -3rem; text-align: center;">
  <h1 style="color: white; margin: 0; font-size: 2.5rem;">AWS Compute Services</h1>
  <p style="font-size: 1.25rem; margin-top: 1rem; opacity: 0.9;">EC2 virtual servers, Lambda serverless functions, Auto Scaling, and container patterns for running your applications in the cloud.</p>
</div>

## Why Compute Services Matter

Every application needs somewhere to run. In traditional IT, that means buying servers, racking them in a data center, and managing hardware failures. AWS compute services let you skip all of that and get straight to running your code.

This guide covers two main approaches:

- **EC2 (Elastic Compute Cloud)**: Virtual servers you control completely, like renting computers in the cloud
- **Lambda**: Serverless functions where you just upload code and AWS handles everything else

**Which should you choose?** Start with EC2 if you need full control, have long-running processes, or are migrating existing applications. Choose Lambda for event-driven workloads, APIs, or when you want zero server management.

---

## AWS Crash Course: From Zero to Cloud in 30 Minutes

If you're completely new to AWS, this crash course will get you up and running quickly. We'll build something real while learning the essentials.

### What You'll Build
In the next 30 minutes, you'll deploy a simple web application that demonstrates core AWS concepts. By the end, you'll have:
- A virtual server running in the cloud (EC2)
- A globally accessible website (S3 + CloudFront)
- A managed database (RDS)
- Monitoring and alerts (CloudWatch)
- All connected securely

### Prerequisites
- An AWS account (sign up at aws.amazon.com - credit card required but we'll stay in free tier)
- Basic command line knowledge
- A web browser
- 30 minutes of focused time

### Step 1: Secure Your Account (5 minutes)
**Never skip this step.** AWS gives you powerful tools that can cost money if misused.

1. **Enable MFA on Root Account**
   ```
   AWS Console → Your Account → Security Credentials → MFA
   ```
   Use Google Authenticator or similar. This prevents unauthorized access even if your password leaks.

2. **Create Your First IAM User**
   ```
   IAM → Users → Add User
   Username: your-name-admin
   Access: AWS Management Console access
   Permissions: AdministratorAccess (for learning only!)
   ```

3. **Set Up Billing Alerts**
   ```
   Billing → Billing Preferences → Receive Billing Alerts
   CloudWatch → Alarms → Create Alarm
   Metric: EstimatedCharges
   Threshold: $5
   ```

**Common Pitfall**: Using root account for daily work. This is like using admin/sudo for everything - dangerous and unnecessary.

### Step 2: Launch Your First Server (10 minutes)

Let's create a virtual server (EC2 instance) - your computer in the cloud.

1. **Navigate to EC2**
   ```
   Services → EC2 → Launch Instance
   ```

2. **Configure Your Instance**
   - **Name**: MyFirstServer
   - **OS**: Amazon Linux 2023 (free tier eligible)
   - **Instance Type**: t2.micro (1 CPU, 1GB RAM - free tier)
   - **Key Pair**: Create new → Download .pem file (guard this carefully!)
   - **Network**: Allow SSH (22) and HTTP (80)

3. **Connect to Your Server**
   ```bash
   # Mac/Linux
   chmod 400 your-key.pem
   ssh -i your-key.pem ec2-user@your-instance-ip

   # Windows (use PuTTY or WSL)
   ```

4. **Install a Web Server**
   ```bash
   sudo yum update -y
   sudo yum install httpd -y
   sudo systemctl start httpd
   sudo systemctl enable httpd
   echo "<h1>Hello from AWS!</h1>" | sudo tee /var/www/html/index.html
   ```

Visit your instance's public IP in a browser - you have a website!

**What Just Happened?**
- You rented a computer in AWS's data center
- You connected to it over the internet
- You installed software just like on any Linux machine
- Your website is now globally accessible

**Common Pitfalls**:
- Forgetting to allow HTTP in security group (your firewall rules)
- Losing your key pair (no key = no access)
- Leaving instances running (they cost money outside free tier hours)

### Step 3: Create Scalable Storage (5 minutes)

EC2 storage disappears when instances terminate. Let's use S3 for permanent storage.

1. **Create an S3 Bucket**
   ```
   S3 → Create Bucket
   Name: my-first-bucket-[random-numbers]
   Region: Same as your EC2
   Block all public access: OFF (for learning only!)
   ```

2. **Upload a File**
   Create a simple HTML file locally:
   ```html
   <!DOCTYPE html>
   <html>
   <head><title>My S3 Site</title></head>
   <body>
     <h1>This website runs from S3!</h1>
     <p>No servers needed.</p>
   </body>
   </html>
   ```
   Upload via console → Make public

3. **Enable Static Website Hosting**
   ```
   Bucket → Properties → Static website hosting → Enable
   Index document: index.html
   ```

Your website now runs without any servers! Access via the S3 website endpoint.

**Why This Matters**: S3 is virtually unlimited storage that's always available. It's perfect for images, videos, backups, and static websites.

### Step 4: Add a Database (5 minutes)

Most applications need to store data. Let's add a managed database.

1. **Create an RDS Instance**
   ```
   RDS → Create Database
   Engine: MySQL
   Template: Free tier
   DB Instance: db.t3.micro
   Username: admin
   Password: [choose-strong-password]
   ```

2. **Configure Access**
   - Same VPC as your EC2
   - Security group: Allow MySQL port (3306) from your EC2 security group

3. **Connect from EC2**
   ```bash
   sudo yum install mysql -y
   mysql -h your-rds-endpoint -u admin -p
   ```

**What's Happening**: AWS manages backups, updates, and availability. You just use the database.

### Step 5: Monitor Everything (5 minutes)

You can't improve what you don't measure. CloudWatch monitors everything.

1. **View EC2 Metrics**
   ```
   EC2 → Your Instance → Monitoring Tab
   ```
   See CPU, network, disk usage

2. **Create an Alarm**
   ```
   CloudWatch → Alarms → Create Alarm
   Select Metric: EC2 → Per-Instance Metrics
   CPU Utilization > 80% for 5 minutes
   Send notification to your email
   ```

3. **View Logs**
   Install CloudWatch agent on EC2:
   ```bash
   sudo yum install amazon-cloudwatch-agent -y
   ```

**Real Impact**: You'll know about problems before your users do.

### Your First Cloud Architecture

Congratulations! You've built a real cloud architecture:
```
Internet → EC2 (Web Server) → RDS (Database)
    ↓
S3 (Static Assets) → CloudFront (CDN)
    ↓
CloudWatch (Monitoring Everything)
```

### Next Steps After the Crash Course

1. **Clean Up** (Important!):
   - Terminate EC2 instance
   - Delete RDS instance
   - Empty and delete S3 bucket
   - This prevents charges

2. **What to Learn Next**:
   - **IAM Roles**: Give EC2 permission to access S3 without keys
   - **Auto Scaling**: Automatically add/remove servers based on load
   - **Load Balancers**: Distribute traffic across multiple servers
   - **VPC**: Create private networks for security

3. **Practice Projects**:
   - Deploy a WordPress blog (EC2 + RDS)
   - Create a serverless API (Lambda + API Gateway)
   - Build a data pipeline (S3 + Lambda + DynamoDB)

### Common Beginner Mistakes to Avoid

1. **The "It Works on My Machine" Trap**
   - Always test in an AWS environment
   - Local and cloud environments differ

2. **The "Infinite Scale" Misconception**
   - Everything has limits (even in cloud)
   - Plan for growth, but start small

3. **The "Set and Forget" Danger**
   - Cloud resources need maintenance
   - Automate everything possible

4. **The "All Public" Security Hole**
   - Start with least privilege
   - Only expose what's necessary

5. **The "Bill Shock" Surprise**
   - Set up billing alerts immediately
   - Understand pricing before using services
   - Use AWS Cost Explorer weekly

This crash course gives you hands-on experience with core AWS services. You've deployed real infrastructure and learned by doing. The rest of this guide builds on these foundations with deeper knowledge and advanced patterns.

## Core Concepts to Master First

Before diving into specific services, let's understand the fundamental concepts that make cloud computing powerful:

### The Cloud Mental Model
Traditional IT requires you to predict capacity, buy hardware, and maintain everything yourself. Cloud computing flips this model - you provision resources on-demand, scale instantly, and let AWS handle the infrastructure complexity.

### Regions and Availability Zones
AWS operates in multiple geographic regions worldwide. Each region contains multiple Availability Zones (AZs) - essentially separate data centers with independent power, cooling, and networking. This geographic distribution is your foundation for building resilient applications that can survive failures.

### The Pay-as-You-Go Model
Unlike traditional IT where you pay upfront for capacity you might not use, AWS charges based on actual consumption. Launch 100 servers for an hour? You pay for 100 server-hours. This model enables experimentation and scaling without massive capital investment.

### Security as a Shared Responsibility
AWS secures the infrastructure (the "security of the cloud"), while you secure your data and applications ("security in the cloud"). Understanding this division helps you build secure systems from day one.

### Identity and Access Management (IAM)
Before creating any resources, understand IAM - it's the foundation of AWS security. IAM controls who can do what in your AWS account. Start with these principles:
- Never use your root account for daily work
- Create individual users with specific permissions
- Use roles for applications, not hardcoded credentials
- Enable MFA (Multi-Factor Authentication) everywhere

## Your First Steps with AWS

Now that you understand the core concepts, let's get practical. AWS can feel overwhelming with its 200+ services, but you don't need to learn them all at once. Here's a progressive path from beginner to advanced cloud architect.

### Getting Your Hands Dirty

Start with the AWS Free Tier - it gives you 12 months of free access to core services with generous limits. This is your playground for learning without worrying about costs.

1. **Create an AWS Account**: Set up billing alerts immediately (even on free tier)
2. **Secure Your Account**: Enable MFA on your root account and create an IAM user for daily work
3. **Launch Your First EC2 Instance**: Think of it as renting a computer in the cloud
4. **Store Files in S3**: Upload some files and understand object storage
5. **Set Up a Simple Website**: Combine EC2 and S3 to host a basic web application

Each step builds on the previous one, gradually introducing you to how AWS services work together.

## Essential Services for Every Developer

Let's explore the services you'll use most often, understanding not just what they do, but why they matter for real applications.

### Compute Services: Your Application's Brain

#### Amazon EC2 - Virtual Servers

EC2 (Elastic Compute Cloud) is like renting computers in the cloud. You choose the operating system, processing power, memory, and storage. Need a small server for testing? Launch a t2.micro. Building a data processing pipeline? Spin up a compute-optimized instance.

**Real-world example**: A startup begins with one EC2 instance running their web application. As traffic grows, they add more instances behind a load balancer. During Black Friday, they scale to 50 instances, then scale back down afterward.

### Choosing the Right Instance Type

Consider the following when selecting an instance type: What is your workload's primary bottleneck? CPU, memory, storage, or network?

| Instance Family | Best For | Example Use Cases | Starting Price* |
|-----------------|----------|-------------------|-----------------|
| **T3/T3a** (Burstable) | Variable workloads with occasional spikes | Development, small websites, microservices | ~$0.01/hr |
| **M5/M6i** (General Purpose) | Balanced compute, memory, networking | Web servers, app servers, small databases | ~$0.10/hr |
| **C5/C6i** (Compute Optimized) | CPU-intensive tasks | Batch processing, gaming servers, scientific modeling | ~$0.09/hr |
| **R5/R6i** (Memory Optimized) | Large datasets in memory | In-memory databases, real-time analytics | ~$0.13/hr |
| **I3/I4i** (Storage Optimized) | High sequential read/write | Data warehousing, distributed file systems | ~$0.16/hr |
| **P4/G5** (Accelerated) | GPU-intensive workloads | Machine learning training, video encoding | ~$1.00/hr+ |

*Prices are approximate for US East, on-demand. Actual prices vary by region and change over time.

### EC2 Pricing Models

You can significantly reduce costs by choosing the right pricing model:

| Model | Savings | Commitment | Best For |
|-------|---------|------------|----------|
| **On-Demand** | Baseline | None | Unpredictable workloads, testing |
| **Reserved Instances** | Up to 72% | 1-3 years | Steady-state production workloads |
| **Savings Plans** | Up to 72% | 1-3 years | Flexible usage across instance types |
| **Spot Instances** | Up to 90% | None (can be interrupted) | Fault-tolerant batch jobs, CI/CD |

### Practical EC2 Patterns

**1. Right-Sizing Instances**

Check your CPU utilization with CloudWatch. If CPU stays below 20% consistently, you are likely overpaying:

```bash
# Check CPU utilization for the past day
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
  --statistics Average --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z --period 3600
```

**When to downsize**: t3.micro to t3.nano saves 50%. m5.large to t3.medium saves 60% for variable workloads.

**2. Spot Instances for Batch Processing**

Spot instances offer up to 90% savings for interruptible workloads. The key is designing for interruption:

```bash
# Request a Spot instance with a max price
aws ec2 request-spot-instances --spot-price "0.10" \
  --instance-count 1 --type "one-time" \
  --launch-specification file://spot-spec.json
```

**When to use Spot**: Batch processing, CI/CD builds, data analysis, any job that can checkpoint and resume.

**3. Auto-Recovery for Critical Instances**

Configure CloudWatch alarms to automatically recover instances that fail system status checks:

```bash
# Set up auto-recovery alarm
aws cloudwatch put-metric-alarm --alarm-name ec2-auto-recovery \
  --metric-name StatusCheckFailed_System --namespace AWS/EC2 \
  --statistic Maximum --period 60 --evaluation-periods 2 \
  --threshold 0 --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:automate:region:ec2:recover
```

### Common EC2 Pitfalls and Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| Zombie Instances | Running instances doing nothing useful | Tag with owner/purpose, run weekly audits |
| Lost SSH Keys | Cannot access instance | Use Systems Manager Session Manager instead |
| Wrong Instance Family | Slower performance than expected | Match instance to workload (see table above) |
| IP Address Changes | Public IP changes after stop/start | Use Elastic IPs for static addresses |
| Unexpected Costs | Bill higher than expected | Set billing alerts, use AWS Cost Explorer |

**Quick fix for lost SSH access**: Use AWS Systems Manager to connect without keys:
```bash
aws ssm start-session --target i-1234567890abcdef0
```

**Quick fix for static IPs**: Allocate an Elastic IP to prevent IP changes:
```bash
aws ec2 allocate-address --domain vpc
aws ec2 associate-address --instance-id i-1234567890abcdef0 --allocation-id eipalloc-xxx
```

#### AWS Lambda - Serverless Computing

Lambda represents a paradigm shift. Instead of managing servers, you upload your code and AWS runs it in response to events. You pay only for the milliseconds your code executes.

**Real-world example**: An e-commerce site uses Lambda to resize product images. When a seller uploads a photo, Lambda automatically creates multiple sizes for different devices. No servers to manage, automatic scaling, and you only pay when images are processed.

### EC2 vs Lambda: When to Use Each

| Factor | EC2 | Lambda |
|--------|-----|--------|
| **Execution time** | Unlimited | Max 15 minutes |
| **Startup time** | Minutes | Milliseconds (cold start: seconds) |
| **Pricing** | Per hour (even when idle) | Per millisecond of execution |
| **Control** | Full OS access | Runtime environment only |
| **Scaling** | Manual or Auto Scaling groups | Automatic, instant |
| **Best for** | Long-running apps, migrations | Event-driven, APIs, batch jobs |

### Practical Lambda Patterns

**1. Event-Driven Processing** (triggered by S3 uploads, database changes, etc.)

```python
def lambda_handler(event, context):
    # Get the uploaded file info from S3 trigger
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Process the file (resize image, parse CSV, etc.)
    # ...
    return {'statusCode': 200, 'body': 'Processed successfully'}
```

**2. API Backend** (with API Gateway)

```python
def lambda_handler(event, context):
    if event['httpMethod'] == 'GET':
        return {'statusCode': 200, 'body': json.dumps({'message': 'Hello'})}
    elif event['httpMethod'] == 'POST':
        data = json.loads(event['body'])
        # Process data...
        return {'statusCode': 201, 'body': json.dumps({'created': True})}
```

**3. Scheduled Tasks** (with CloudWatch Events/EventBridge)

```python
def lambda_handler(event, context):
    # Runs on schedule (e.g., daily cleanup, reports)
    # Delete old files, send notifications, generate reports
    return {'statusCode': 200, 'body': 'Task completed'}
```

### Common Lambda Pitfalls and Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| Cold Starts | First invocation takes 3-5 seconds | Use Provisioned Concurrency or keep warm with scheduled pings |
| 15-Minute Timeout | Long processes fail | Break into smaller functions, use Step Functions |
| Out of Memory | Function crashes on large files | Stream data instead of loading all at once |
| Package Too Large | Cannot upload (>50MB limit) | Use Lambda Layers or container images (up to 10GB) |
| High Costs | Unexpected bills | Optimize memory settings, reduce execution time |

**Reducing cold starts**: Keep deployment packages small. Avoid large libraries like pandas (40MB) when simpler alternatives work.

**Handling long processes**: Use AWS Step Functions to orchestrate multiple Lambda functions:

```json
{
  "StartAt": "ProcessChunk",
  "States": {
    "ProcessChunk": {
      "Type": "Map",
      "ItemsPath": "$.chunks",
      "MaxConcurrency": 10,
      "Iterator": { "StartAt": "DoWork", "States": {...} }
    }
  }
}
```

**Handling large files**: Stream instead of loading everything into memory:

```python
# Stream large files line by line
for line in s3_object['Body'].iter_lines():
    process_line(line)  # Process one line at a time
```

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Storage Services](storage.html) - S3 and EBS for your data
- [Database Services](databases.html) - RDS and DynamoDB
- [Networking](../networking.html) - VPC and load balancers
- [Security](security.html) - IAM and security best practices
- [Infrastructure & Operations](infrastructure.html) - CloudFormation, monitoring, and cost optimization
- [Kubernetes on AWS](../kubernetes/) - Container orchestration with EKS
- [Docker](../docker/) - Containerization fundamentals
