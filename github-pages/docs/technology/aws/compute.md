---
layout: docs
title: AWS Compute Services
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "server"
---

# AWS Compute Services

This guide covers AWS compute services including EC2 virtual servers and Lambda serverless functions. Start here if you're new to AWS - the crash course will have you deploying your first cloud application in 30 minutes.

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

**Practical EC2 Patterns**:

1. **Right-Sizing Instances**
   ```bash
   # Check current utilization
   aws cloudwatch get-metric-statistics \
     --namespace AWS/EC2 \
     --metric-name CPUUtilization \
     --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
     --statistics Average \
     --start-time 2024-01-01T00:00:00Z \
     --end-time 2024-01-02T00:00:00Z \
     --period 3600
   ```
   
   If CPU < 20% consistently, you're overpaying. Consider:
   - t3.micro → t3.nano (save 50%)
   - m5.large → t3.medium (save 60% for variable workloads)

2. **Spot Instances for Batch Processing**
   ```python
   # Example: Processing logs with 90% cost savings
   import boto3
   
   ec2 = boto3.client('ec2')
   
   # Request Spot instance
   response = ec2.request_spot_instances(
       SpotPrice='0.10',  # Max price you'll pay
       InstanceCount=1,
       LaunchSpecification={
           'ImageId': 'ami-12345678',
           'InstanceType': 'c5.large',
           'UserData': base64.b64encode('''#!/bin/bash
               aws s3 cp s3://my-logs/raw/ /tmp/logs/ --recursive
               /opt/process-logs.sh
               aws s3 cp /tmp/results/ s3://my-logs/processed/ --recursive
               shutdown -h now  # Self-terminate to save money
           '''.encode()).decode()
       }
   )
   ```

3. **Auto-Recovery for Critical Instances**
   ```bash
   # Create CloudWatch alarm for auto-recovery
   aws cloudwatch put-metric-alarm \
     --alarm-name ec2-recovery-i-1234567890abcdef0 \
     --alarm-description "Recover instance when it fails" \
     --metric-name StatusCheckFailed_System \
     --namespace AWS/EC2 \
     --statistic Maximum \
     --period 60 \
     --evaluation-periods 2 \
     --threshold 0 \
     --comparison-operator GreaterThanThreshold \
     --alarm-actions arn:aws:automate:region:ec2:recover
   ```

**Common EC2 Pitfalls and Solutions**:

1. **The "Zombie Instance" Problem**
   - **Symptom**: Instances running but not doing anything useful
   - **Solution**: Tag instances with purpose and owner, run weekly audits
   ```bash
   # Find instances without Owner tag
   aws ec2 describe-instances \
     --query 'Reservations[].Instances[?!Tags[?Key==`Owner`]].[InstanceId,State.Name]' \
     --output table
   ```

2. **The "SSH Key Lost Forever" Disaster**
   - **Symptom**: Can't access instance after losing private key
   - **Prevention**: Use Systems Manager Session Manager
   ```bash
   # Connect without SSH keys
   aws ssm start-session --target i-1234567890abcdef0
   ```

3. **The "Wrong Instance Family" Performance Issue**
   - **Symptom**: Application runs slower than expected
   - **Solution**: Match instance to workload
   - CPU-intensive → C5 (compute optimized)
   - Memory-intensive → R5 (memory optimized)  
   - Balanced → M5 (general purpose)
   - Bursty → T3 (burstable performance)

4. **The "Public IP Changed" Confusion**
   - **Symptom**: IP address changes after stop/start
   - **Solution**: Use Elastic IPs for static addresses
   ```bash
   # Allocate and associate Elastic IP
   ALLOC_ID=$(aws ec2 allocate-address --query 'AllocationId' --output text)
   aws ec2 associate-address --instance-id i-1234567890abcdef0 --allocation-id $ALLOC_ID
   ```

#### AWS Lambda - Serverless Computing
Lambda represents a paradigm shift. Instead of managing servers, you upload your code and AWS runs it in response to events. You pay only for the milliseconds your code executes.

**Real-world example**: An e-commerce site uses Lambda to resize product images. When a seller uploads a photo, Lambda automatically creates multiple sizes for different devices. No servers to manage, automatic scaling, and you only pay when images are processed.

**Practical Lambda Patterns**:

1. **Event-Driven Image Processing**
   ```python
   import json
   import boto3
   from PIL import Image
   import io
   
   s3 = boto3.client('s3')
   
   def lambda_handler(event, context):
       # Triggered by S3 upload
       bucket = event['Records'][0]['s3']['bucket']['name']
       key = event['Records'][0]['s3']['object']['key']
       
       # Download image
       response = s3.get_object(Bucket=bucket, Key=key)
       img = Image.open(io.BytesIO(response['Body'].read()))
       
       # Create thumbnails
       sizes = [(128, 128), (256, 256), (512, 512)]
       for size in sizes:
           resized = img.resize(size, Image.LANCZOS)
           
           # Save to S3
           buffer = io.BytesIO()
           resized.save(buffer, 'JPEG')
           buffer.seek(0)
           
           new_key = f"thumbnails/{size[0]}x{size[1]}/{key}"
           s3.put_object(Bucket=bucket, Key=new_key, Body=buffer.getvalue())
       
       return {'statusCode': 200, 'body': json.dumps('Thumbnails created')}
   ```

2. **API Backend Without Servers**
   ```python
   # Lambda function for REST API
   import json
   import boto3
   from decimal import Decimal
   
   dynamodb = boto3.resource('dynamodb')
   table = dynamodb.Table('Users')
   
   def lambda_handler(event, context):
       http_method = event['httpMethod']
       
       if http_method == 'GET':
           # Get user by ID
           user_id = event['pathParameters']['id']
           response = table.get_item(Key={'userId': user_id})
           
           return {
               'statusCode': 200,
               'headers': {'Content-Type': 'application/json'},
               'body': json.dumps(response.get('Item', {}), default=str)
           }
       
       elif http_method == 'POST':
           # Create new user
           body = json.loads(event['body'])
           table.put_item(Item=body)
           
           return {
               'statusCode': 201,
               'headers': {'Content-Type': 'application/json'},
               'body': json.dumps({'message': 'User created'})
           }
   ```

3. **Scheduled Data Processing**
   ```python
   # CloudWatch Events triggers this daily
   def lambda_handler(event, context):
       # Clean up old data
       s3 = boto3.client('s3')
       
       # List objects older than 30 days
       response = s3.list_objects_v2(Bucket='my-temp-bucket')
       
       for obj in response.get('Contents', []):
           if (datetime.now(timezone.utc) - obj['LastModified']).days > 30:
               s3.delete_object(Bucket='my-temp-bucket', Key=obj['Key'])
               print(f"Deleted: {obj['Key']}")
   ```

**Common Lambda Pitfalls and Solutions**:

1. **The "Cold Start" Surprise**
   - **Symptom**: First invocation takes 3-5 seconds
   - **Solutions**:
     ```python
     # 1. Provisioned Concurrency (costs money)
     # 2. Keep Lambda warm with scheduled pings
     # 3. Optimize package size
     
     # Bad: Large deployment package
     # requirements.txt
     pandas==1.5.0  # 40MB
     numpy==1.23.0  # 20MB
     
     # Good: Use Lambda Layers or lighter alternatives
     # Use AWS SDK (boto3) - pre-installed
     # Use json instead of pandas for simple data
     ```

2. **The "15-Minute Timeout" Wall**
   - **Symptom**: Function times out for long processes (Lambda max timeout is 15 minutes)
   - **Solution**: Break into smaller functions or use Step Functions
   ```python
   # Step Functions state machine for long workflows
   {
     "Comment": "Process large dataset",
     "StartAt": "SplitData",
     "States": {
       "SplitData": {
         "Type": "Task",
         "Resource": "arn:aws:lambda:region:account:function:split-data",
         "Next": "ProcessInParallel"
       },
       "ProcessInParallel": {
         "Type": "Map",
         "ItemsPath": "$.chunks",
         "MaxConcurrency": 10,
         "Iterator": {
           "StartAt": "ProcessChunk",
           "States": {
             "ProcessChunk": {
               "Type": "Task",
               "Resource": "arn:aws:lambda:region:account:function:process-chunk",
               "End": true
             }
           }
         },
         "Next": "MergeResults"
       }
     }
   }
   ```

3. **The "Out of Memory" Crisis**
   - **Symptom**: Function fails with memory errors
   - **Solution**: Stream data instead of loading all at once
   ```python
   # Bad: Loading entire file
   def lambda_handler(event, context):
       s3 = boto3.client('s3')
       obj = s3.get_object(Bucket='bucket', Key='huge-file.csv')
       data = obj['Body'].read()  # Boom! Out of memory
   
   # Good: Stream processing
   def lambda_handler(event, context):
       s3 = boto3.client('s3')
       obj = s3.get_object(Bucket='bucket', Key='huge-file.csv')
       
       for line in obj['Body'].iter_lines():
           process_line(line)  # Process one line at a time
   ```

4. **The "Deployment Package Too Large" Block**
   - **Symptom**: Cannot upload function (>50MB zipped)
   - **Solution**: Use Layers and container images
   ```bash
   # Create a Lambda Layer for dependencies
   pip install -r requirements.txt -t python/
   zip -r layer.zip python/
   
   aws lambda publish-layer-version \
     --layer-name my-dependencies \
     --zip-file fileb://layer.zip \
     --compatible-runtimes python3.9
   
   # Or use container images (up to 10GB)
   FROM public.ecr.aws/lambda/python:3.9
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY app.py .
   CMD ["app.lambda_handler"]
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
