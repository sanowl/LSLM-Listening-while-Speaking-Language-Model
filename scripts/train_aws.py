#!/usr/bin/env python3
"""AWS Training script for LSLM model using SageMaker."""

import argparse
import os
from datetime import datetime

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSLM model on AWS")
    parser.add_argument("--aws-profile", type=str, default="default", help="AWS profile name")
    parser.add_argument("--region", type=str, default="us-west-2", help="AWS region")
    parser.add_argument("--instance-type", type=str, default="ml.p3.2xlarge", help="SageMaker instance type")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket for data and models")
    parser.add_argument("--data-prefix", type=str, required=True, help="S3 prefix for data")
    parser.add_argument("--output-prefix", type=str, required=True, help="S3 prefix for output")
    return parser.parse_args()

def setup_sagemaker(args):
    """Setup SageMaker session and role."""
    session = boto3.Session(profile_name=args.aws_profile, region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=session)
    
    # Get SageMaker execution role
    role = sagemaker.get_execution_role()
    return sagemaker_session, role

def create_estimator(sagemaker_session, role, args):
    """Create PyTorch estimator for training."""
    return PyTorch(
        entry_point="src/lslm/train.py",
        source_dir=".",
        role=role,
        framework_version="1.8.1",
        py_version="py36",
        instance_count=1,
        instance_type=args.instance_type,
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "data_dir": "/opt/ml/input/data/training",
            "output_dir": "/opt/ml/model",
        },
        environment={
            "PYTHONPATH": "/opt/ml/code"
        }
    )

def prepare_data_channels(args):
    """Prepare data channels for training."""
    return {
        "training": f"s3://{args.bucket}/{args.data_prefix}/train",
        "validation": f"s3://{args.bucket}/{args.data_prefix}/val"
    }

def main():
    args = parse_args()
    
    # Setup SageMaker
    sagemaker_session, role = setup_sagemaker(args)
    
    # Create unique job name
    job_name = f"lslm-training-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    # Create estimator
    estimator = create_estimator(sagemaker_session, role, args)
    
    # Prepare data channels
    data_channels = prepare_data_channels(args)
    
    # Start training
    print(f"Starting training job: {job_name}")
    estimator.fit(
        inputs=data_channels,
        job_name=job_name,
        wait=True
    )
    
    # Save model artifacts
    model_artifact = estimator.model_data
    print(f"Training completed. Model artifacts saved to: {model_artifact}")
    
    # Copy model artifacts to specified output location
    output_path = f"s3://{args.bucket}/{args.output_prefix}/{job_name}"
    s3 = boto3.client("s3")
    s3.copy(
        {"Bucket": model_artifact.split("/")[2], "Key": "/".join(model_artifact.split("/")[3:])},
        args.bucket,
        f"{args.output_prefix}/{job_name}/model.tar.gz"
    )
    print(f"Model copied to: {output_path}")

if __name__ == "__main__":
    main() 