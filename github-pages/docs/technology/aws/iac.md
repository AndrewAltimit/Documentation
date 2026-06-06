---
layout: docs
title: "AWS Infrastructure as Code"
permalink: /docs/technology/aws/iac.html
hide_title: true
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "server"
---

The biggest shift in cloud operations is treating infrastructure like software. Instead of clicking through the console, you declare what you want in code, review it like any other change, and deploy it repeatably. This page covers CloudFormation (AWS's native templating engine), how to choose among CloudFormation, Terraform, and the CDK, and a full AWS CDK microservices stack.

---

## AWS CloudFormation - Infrastructure as Code
CloudFormation lets you define your entire infrastructure in JSON or YAML templates. Instead of clicking through the console, you describe what you want and CloudFormation builds it.

**Why CloudFormation Matters**:
- Version control your infrastructure
- Replicate environments exactly
- Roll back changes if something breaks
- Share templates with your team

**Practical CloudFormation Example**:
```yaml
# template.yml - Complete web application stack
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Web application with auto-scaling'

Parameters:
  KeyName:
    Type: AWS::EC2::KeyPair::KeyName
    Description: EC2 Key Pair for SSH access

Resources:
  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref LoadBalancerSecurityGroup

  # Auto Scaling Group
  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      MinSize: 2
      MaxSize: 10
      DesiredCapacity: 4
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      TargetGroupARNs:
        - !Ref TargetGroup
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300

  # Scaling Policy
  ScaleUpPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AutoScalingGroupName: !Ref AutoScalingGroup
      PolicyType: TargetTrackingScaling
      TargetTrackingConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ASGAverageCPUUtilization
        TargetValue: 70

Outputs:
  LoadBalancerDNS:
    Description: DNS name of load balancer
    Value: !GetAtt LoadBalancer.DNSName
    Export:
      Name: !Sub ${AWS::StackName}-LoadBalancer-DNS
```

**Deploy with**:
```bash
aws cloudformation create-stack \
  --stack-name my-web-app \
  --template-body file://template.yml \
  --parameters ParameterKey=KeyName,ParameterValue=my-key
```

---

## Infrastructure as Code: Never Click Again

### Why Infrastructure as Code Changes Everything

Defining infrastructure in code rather than console clicks unlocks version control, peer review, and automated deployment:

| | Console clicks | Infrastructure as Code |
|---|---|---|
| **Source of truth** | A wiki nobody updates | Configuration files in version control |
| **Reproducibility** | Hope you can recreate it elsewhere | Deploy identical environments with one command |
| **Change safety** | Fear of breaking production | Diff, review, and test in staging first |
| **History** | None | Version control shows exactly what changed and when |

### Choosing Your IaC Tool

| Tool | Pros | Cons | Best for |
|------|------|------|----------|
| **CloudFormation** (AWS native) | Deep AWS integration, no extra tools | Verbose syntax, AWS-only | Teams fully committed to AWS |
| **Terraform** (multi-cloud) | Works across providers, huge community | Must learn HCL | Multi-cloud or teams wanting flexibility |
| **AWS CDK** (developer-friendly) | Define infra in Python/TypeScript with loops and abstractions | Newer, smaller community | Dev teams reusing existing language skills |

### Real-World IaC Evolution

A startup's infrastructure journey:

1. **Month 1**: Everything created via console clicks
2. **Month 3**: Production breaks, nobody remembers how to rebuild
3. **Month 4**: Team adopts Terraform, documents existing infrastructure
4. **Month 6**: All changes go through pull requests
5. **Year 1**: Disaster recovery test - entire production rebuilt in 30 minutes

### Advanced Patterns That Save Your Sanity

```python
from aws_cdk import (
    core as cdk,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_rds as rds,
    aws_secretsmanager as sm,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    custom_resources as cr
)
from constructs import Construct
import json

class MicroservicesStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create VPC with custom configuration
        vpc = ec2.Vpc(
            self, "MicroservicesVPC",
            max_azs=3,
            nat_gateways=2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.ISOLATED,
                    cidr_mask=24
                )
            ]
        )

        # Create ECS Cluster with capacity providers
        cluster = ecs.Cluster(
            self, "Cluster",
            vpc=vpc,
            container_insights=True
        )

        # Add Fargate Spot capacity provider
        cluster.add_capacity_provider(
            ecs.FargateCapacityProvider(
                self, "FargateSpotProvider",
                spot=True
            )
        )

        # Create RDS Aurora Serverless v2
        db_secret = sm.Secret(
            self, "DBSecret",
            generate_secret_string=sm.SecretStringGenerator(
                secret_string_template=json.dumps({"username": "admin"}),
                generate_string_key="password",
                exclude_characters=" %+~`#$&*()|[]{}:;<>?!'/\\"
            )
        )

        db_cluster = rds.DatabaseCluster(
            self, "AuroraCluster",
            engine=rds.DatabaseClusterEngine.aurora_mysql(
                # Use a current Aurora MySQL 3.x version label; this is an
                # example — pin to a version supported in your account/region.
                version=rds.AuroraMysqlEngineVersion.VER_3_04_0
            ),
            serverless_v2_scaling_configuration=rds.ServerlessV2ScalingConfiguration(
                min_capacity=0.5,
                max_capacity=2
            ),
            credentials=rds.Credentials.from_secret(db_secret),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.ISOLATED
            ),
            backup=rds.BackupProps(
                retention=cdk.Duration.days(7)
            ),
            deletion_protection=True
        )

        # Create shared ALB
        alb = elbv2.ApplicationLoadBalancer(
            self, "ALB",
            vpc=vpc,
            internet_facing=True,
            http2_enabled=True
        )

        # Add CloudWatch alarms
        alarm = cloudwatch.Alarm(
            self, "HighErrorRate",
            metric=alb.metric_target_response_time(),
            threshold=1000,
            evaluation_periods=2
        )

        # SNS topic for alarms
        alarm_topic = sns.Topic(
            self, "AlarmTopic",
            display_name="Microservices Alarms"
        )

        alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))

        # Deploy microservices
        self.deploy_microservice(
            cluster=cluster,
            alb=alb,
            service_name="users",
            image="users-service:latest",
            port=8080,
            priority=1,
            path_pattern="/users/*",
            environment={
                "DB_SECRET_ARN": db_secret.secret_arn,
                "DB_CLUSTER_ARN": db_cluster.cluster_arn
            }
        )

        self.deploy_microservice(
            cluster=cluster,
            alb=alb,
            service_name="orders",
            image="orders-service:latest",
            port=8081,
            priority=2,
            path_pattern="/orders/*",
            environment={
                "DB_SECRET_ARN": db_secret.secret_arn,
                "DB_CLUSTER_ARN": db_cluster.cluster_arn
            }
        )

        # Create API Gateway for serverless endpoints
        api = apigw.RestApi(
            self, "MicroservicesAPI",
            deploy_options=apigw.StageOptions(
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=True,
                tracing_enabled=True
            )
        )

        # Lambda function for async processing
        async_processor = lambda_.Function(
            self, "AsyncProcessor",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda"),
            vpc=vpc,
            environment={
                "DB_SECRET_ARN": db_secret.secret_arn
            },
            reserved_concurrent_executions=100,
            tracing=lambda_.Tracing.ACTIVE
        )

        # Grant permissions
        db_secret.grant_read(async_processor)
        db_cluster.grant_connect(async_processor)

        # Custom resource for database initialization
        db_init = cr.AwsCustomResource(
            self, "DBInit",
            on_create=cr.AwsSdkCall(
                service="RDS",
                action="executeStatement",
                parameters={
                    "resourceArn": db_cluster.cluster_arn,
                    "secretArn": db_secret.secret_arn,
                    "database": "mysql",
                    "sql": "CREATE DATABASE IF NOT EXISTS microservices;"
                },
                physical_resource_id=cr.PhysicalResourceId.of("DBInit")
            ),
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=[db_cluster.cluster_arn]
            )
        )

        # Output values
        cdk.CfnOutput(
            self, "ALBDNSName",
            value=alb.load_balancer_dns_name,
            description="ALB DNS Name"
        )

        cdk.CfnOutput(
            self, "APIEndpoint",
            value=api.url,
            description="API Gateway Endpoint"
        )

    def deploy_microservice(self,
                           cluster: ecs.Cluster,
                           alb: elbv2.ApplicationLoadBalancer,
                           service_name: str,
                           image: str,
                           port: int,
                           priority: int,
                           path_pattern: str,
                           environment: dict):
        """Deploy a microservice to ECS"""

        # Create task definition
        task_definition = ecs.FargateTaskDefinition(
            self, f"{service_name}TaskDef",
            memory_limit_mib=512,
            cpu=256
        )

        # Add container
        container = task_definition.add_container(
            f"{service_name}Container",
            image=ecs.ContainerImage.from_registry(image),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=service_name
            ),
            environment=environment,
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", f"curl -f http://localhost:{port}/health || exit 1"],
                interval=cdk.Duration.seconds(30),
                timeout=cdk.Duration.seconds(5),
                retries=3
            )
        )

        container.add_port_mappings(
            ecs.PortMapping(
                container_port=port,
                protocol=ecs.Protocol.TCP
            )
        )

        # Create service
        service = ecs.FargateService(
            self, f"{service_name}Service",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=2,
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(
                    capacity_provider="FARGATE_SPOT",
                    weight=2
                ),
                ecs.CapacityProviderStrategy(
                    capacity_provider="FARGATE",
                    weight=1
                )
            ],
            circuit_breaker=ecs.DeploymentCircuitBreaker(
                rollback=True
            )
        )

        # Configure auto-scaling
        scaling = service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=10
        )

        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=cdk.Duration.seconds(60),
            scale_out_cooldown=cdk.Duration.seconds(60)
        )

        scaling.scale_on_request_count(
            "RequestScaling",
            requests_per_target=1000,
            target_group=alb.add_targets(
                f"{service_name}TG",
                port=port,
                targets=[service],
                health_check=elbv2.HealthCheck(
                    path=f"/{service_name}/health",
                    interval=cdk.Duration.seconds(30)
                )
            )
        )

        # Add ALB listener rule
        alb.add_listener(
            f"{service_name}Listener",
            port=80
        ).add_targets(
            f"{service_name}Targets",
            port=port,
            targets=[service],
            priority=priority,
            conditions=[
                elbv2.ListenerCondition.path_patterns([path_pattern])
            ]
        )
```

---

## Key Takeaways

- **Codify everything.** CloudFormation, CDK, or Terraform turn clicks into reviewable, repeatable code. Manual console changes drift and cannot be reproduced.
- **Pick the tool that fits the team.** CloudFormation for AWS-native simplicity, Terraform for multi-cloud, CDK to define infrastructure in a real programming language with loops and abstractions.
- **Review infrastructure like code.** Send every change through pull requests and test in staging. The payoff is a disaster-recovery story measured in minutes, not days.

---

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Architecture Patterns & Case Studies](architecture.html) - Where these stacks fit
- [Cost Optimization](cost.html) - IaC-driven budgets and Spot management
- [Terraform](../terraform/) - Alternative multi-cloud IaC approach
- [Compute Services](compute.html) - EC2, Lambda, and container patterns
