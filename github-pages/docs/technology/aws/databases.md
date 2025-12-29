---
layout: docs
title: AWS Database Services
toc: true
toc_sticky: true
toc_label: "On This Page"
toc_icon: "database"
---

# AWS Database Services

## Why Managed Databases Matter

Running databases yourself means handling backups, patching, replication, failover, and monitoring. AWS managed databases handle all of that, letting you focus on your application instead of database administration.

Consider the following when choosing a database: What type of data will you store? How will you query it? What scale do you need? How important is consistency vs. performance?

---

## Choosing the Right Database

Before diving into specific services, understand the two main categories:

| Category | When to Use | AWS Services |
|----------|-------------|--------------|
| **Relational (SQL)** | Structured data, complex queries, transactions | RDS, Aurora |
| **NoSQL** | Flexible schemas, massive scale, simple queries | DynamoDB, DocumentDB, ElastiCache |

### Quick Decision Guide

| Your Situation | Recommended Service |
|----------------|---------------------|
| Traditional web app with complex queries | RDS (PostgreSQL or MySQL) |
| Need maximum SQL performance | Aurora |
| High-scale with simple key-value access | DynamoDB |
| Need caching layer | ElastiCache (Redis) |
| Document storage (MongoDB compatible) | DocumentDB |
| Time-series data | Timestream |
| Graph relationships | Neptune |

---

## Amazon RDS - Managed Relational Databases

RDS runs traditional databases (MySQL, PostgreSQL, MariaDB, Oracle, SQL Server) but handles the operational burden: backups, patching, replication, and failover.

**Real-world example**: A SaaS application uses RDS PostgreSQL for customer data. RDS automatically backs up the database nightly, replicates to a standby instance for high availability, and can scale up during busy periods.

### When to Use RDS

- You need SQL queries with JOINs across tables
- Your data has clear relationships (users, orders, products)
- You need ACID transactions (banking, inventory)
- Your team knows SQL and relational databases

### RDS Instance Classes

| Class | Best For | vCPUs | Memory |
|-------|----------|-------|--------|
| **db.t3** | Development, small workloads | 2-8 | 1-32 GB |
| **db.m5** | General production workloads | 2-96 | 8-384 GB |
| **db.r5** | Memory-intensive (large datasets) | 2-96 | 16-768 GB |

---

## Amazon Aurora - High-Performance Relational

Aurora is AWS's cloud-native database, compatible with MySQL and PostgreSQL. It offers up to 5x the throughput of standard MySQL and 3x that of standard PostgreSQL, with automatic storage scaling up to 128 TB.

### When to Use Aurora vs RDS

| Factor | RDS | Aurora |
|--------|-----|--------|
| **Cost** | Lower for small workloads | Better value at scale |
| **Performance** | Standard | 3-5x faster |
| **Storage** | Manual provisioning | Auto-scales to 128 TB |
| **Replicas** | Up to 5 | Up to 15 with faster failover |
| **Best for** | Dev/test, small production | High-performance production |

---

## Amazon DynamoDB - NoSQL at Scale

DynamoDB is a key-value and document database designed for applications needing consistent single-digit millisecond performance at any scale. It handles millions of requests per second without capacity planning.

**Real-world example**: A mobile game uses DynamoDB to store player profiles and game state. Whether 100 or 10 million players are online, DynamoDB maintains consistent performance.

### When to Use DynamoDB

- Simple access patterns (get item by key, query by partition)
- Massive scale requirements
- Need single-digit millisecond latency
- Schema may evolve over time

### When NOT to Use DynamoDB

- Complex queries with multiple JOINs
- Ad-hoc reporting and analytics
- Need for transactions across many items (limited support)
- Team unfamiliar with NoSQL patterns

---

## Amazon ElastiCache - In-Memory Caching

ElastiCache provides Redis and Memcached for caching frequently accessed data. Adding a cache layer can reduce database load by 90% and improve response times from milliseconds to microseconds.

### Common Caching Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Cache-aside** | Read-heavy workloads | Check cache first, then database |
| **Write-through** | Need cache consistency | Write to cache and database together |
| **Session storage** | Web applications | Store user sessions in Redis |
| **Rate limiting** | API protection | Track request counts per user |

---

## DynamoDB Best Practices

### Key Design Principles

DynamoDB works differently from relational databases. Success depends on understanding these principles:

**1. Design for your access patterns first**: Unlike SQL where you model data then write queries, DynamoDB requires knowing your queries upfront.

**2. Use composite keys**: Combine partition key (PK) and sort key (SK) to enable efficient queries.

**3. Denormalize data**: Store related data together. It is okay to duplicate data across items.

### Single Table Design

Advanced DynamoDB usage puts all entity types in one table, using different key patterns:

| Entity | PK | SK | Example |
|--------|----|----|---------|
| User | `USER#123` | `PROFILE` | User profile data |
| User's orders | `USER#123` | `ORDER#2024-01-15` | Order summary |
| Order details | `ORDER#456` | `DETAIL` | Full order data |

This enables fetching a user and their recent orders in a single query.

### DynamoDB Pricing Models

| Mode | Best For | How It Works |
|------|----------|--------------|
| **On-Demand** | Variable or unpredictable workloads | Pay per request, no capacity planning |
| **Provisioned** | Steady, predictable workloads | Set read/write capacity, lower cost |

**Start with On-Demand** for new applications. Switch to Provisioned once you understand your traffic patterns.

### Essential DynamoDB Commands

```bash
# Create a table with on-demand billing
aws dynamodb create-table --table-name MyTable \
  --attribute-definitions AttributeName=PK,AttributeType=S \
  --key-schema AttributeName=PK,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# Put an item
aws dynamodb put-item --table-name MyTable \
  --item '{"PK": {"S": "USER#123"}, "name": {"S": "Alice"}}'

# Query items by partition key
aws dynamodb query --table-name MyTable \
  --key-condition-expression "PK = :pk" \
  --expression-attribute-values '{":pk": {"S": "USER#123"}}'
```

---

## Database Selection Summary

| Use Case | Service | Why | Cost Consideration |
|----------|---------|-----|-------------------|
| Traditional web apps | RDS (PostgreSQL/MySQL) | SQL, JOINs, transactions | Pay for instance size |
| High-performance SQL | Aurora | 3-5x faster than RDS | Higher base cost, better at scale |
| Massive scale, simple queries | DynamoDB | Consistent millisecond latency | Pay per request or capacity |
| Caching layer | ElastiCache Redis | Sub-millisecond reads | Pay for node size |
| Time-series (IoT, metrics) | Timestream | Built-in time functions | Pay for writes and queries |
| Graph data (social, fraud) | Neptune | Native graph traversal | Pay for instance size |
| MongoDB workloads | DocumentDB | MongoDB-compatible | Pay for instance size |

### Common Patterns

- **Web application**: RDS/Aurora for main data + ElastiCache for sessions and hot data
- **Mobile app**: DynamoDB for user data + S3 for media
- **Analytics**: DynamoDB/RDS for operational data, replicated to S3 for analytics with Athena

---

## See Also

- [AWS Hub](./) - Overview of all AWS documentation
- [Compute Services](compute.html) - Lambda with DynamoDB patterns
- [Storage Services](storage.html) - S3 for database backups
- [Security](security.html) - Database encryption and access control
- [Infrastructure & Operations](infrastructure.html) - Database IaC and monitoring
- [Database Design Guide](../database-design.html) - General database concepts
