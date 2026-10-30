"""A least-privilege IAM policy for people and CI that use Skyhammer (SE-4).

``user_policy(account_id)`` covers everything the SDK and CLI do, scoped to what Skyhammer
creates: the ``skyhammer-<account>-<region>`` buckets, ``skyhammer-grader-*`` Lambdas,
the ``skyhammer-bootstrap`` stack and its roles, and this account's Bedrock and SageMaker
resources. Attach it to a role or group; ``skyhammer auth policy`` prints it.

Calling models is account-wide by nature (any base model or profile you choose), so
Bedrock invoke actions use model and profile ARN patterns rather than a name prefix.
Pass ``bootstrap=False`` to leave out stack management for people who only use an
already-bootstrapped account.
"""

from __future__ import annotations

from typing import Any

BOOTSTRAP_STACK = "skyhammer-bootstrap"


def user_policy(
    account_id: str, *, partition: str = "aws", bootstrap: bool = True
) -> dict[str, Any]:
    acct = account_id
    arn = f"arn:{partition}"

    def bedrock(resource: str, region: str = "*", account: str = acct) -> str:
        return f"{arn}:bedrock:{region}:{account}:{resource}"

    def sagemaker(resource: str) -> str:
        return f"{arn}:sagemaker:*:{acct}:{resource}"

    statements: list[dict[str, Any]] = [
        {
            "Sid": "CallModels",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:ApplyGuardrail",
                "bedrock:GetInferenceProfile",
                "bedrock:Rerank",
            ],
            "Resource": [
                f"{arn}:bedrock:*::foundation-model/*",
                bedrock("inference-profile/*"),
                bedrock("application-inference-profile/*"),
                bedrock("custom-model/*"),
                bedrock("custom-model-deployment/*"),
                bedrock("imported-model/*"),
                bedrock("guardrail/*"),
            ],
        },
        {
            "Sid": "CustomizeServeEvaluate",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelCustomizationJob",
                "bedrock:StopModelCustomizationJob",
                "bedrock:CreateModelImportJob",
                "bedrock:CreateCustomModelDeployment",
                "bedrock:DeleteCustomModelDeployment",
                "bedrock:DeleteCustomModel",
                "bedrock:DeleteImportedModel",
                "bedrock:CreateModelInvocationJob",
                "bedrock:StopModelInvocationJob",
                "bedrock:CreateEvaluationJob",
                "bedrock:StopEvaluationJob",
                "bedrock:CreateGuardrail",
                "bedrock:DeleteGuardrail",
                "bedrock:TagResource",
            ],
            "Resource": [
                bedrock("model-customization-job/*"),
                bedrock("model-import-job/*"),
                bedrock("custom-model/*"),
                bedrock("custom-model-deployment/*"),
                bedrock("imported-model/*"),
                bedrock("model-invocation-job/*"),
                bedrock("evaluation-job/*"),
                bedrock("guardrail/*"),
                f"{arn}:bedrock:*::foundation-model/*",
                bedrock("inference-profile/*"),
            ],
        },
        {
            "Sid": "ReadBedrock",
            "Effect": "Allow",
            "Action": ["bedrock:Get*", "bedrock:List*"],
            "Resource": "*",
        },
        {
            "Sid": "OpenAICompatibleApis",  # bedrock-mantle: open-weight RFT and API keys
            "Effect": "Allow",
            "Action": [
                "bedrock-mantle:CreateInference",
                "bedrock-mantle:CreateFineTuningJob",
                "bedrock-mantle:CancelFineTuningJob",
                "bedrock-mantle:CreateFile",
                "bedrock-mantle:DeleteFile",
                "bedrock-mantle:DeleteModel",
                "bedrock-mantle:Get*",
                "bedrock-mantle:List*",
            ],
            "Resource": f"{arn}:bedrock-mantle:*:{acct}:*",
        },
        {
            "Sid": "BearerTokens",  # short-term Bedrock API keys from these credentials
            "Effect": "Allow",
            "Action": ["bedrock:CallWithBearerToken", "bedrock-mantle:CallWithBearerToken"],
            "Resource": "*",
        },
        {
            "Sid": "SageMakerTrainingAndEndpoints",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:CreateModel",
                "sagemaker:DeleteModel",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:CreateEndpoint",
                "sagemaker:DeleteEndpoint",
                "sagemaker:CreateInferenceComponent",
                "sagemaker:DeleteInferenceComponent",
                "sagemaker:UpdateInferenceComponentRuntimeConfig",
                "sagemaker:InvokeEndpoint",
                "sagemaker:InvokeEndpointWithResponseStream",
                "sagemaker:CreateModelPackageGroup",
                "sagemaker:CreateHub",
                "sagemaker:ImportHubContent",
                "sagemaker:DeleteHubContent",
                "sagemaker:AddTags",
            ],
            "Resource": [
                sagemaker("training-job/*"),
                sagemaker("model/*"),
                sagemaker("endpoint-config/*"),
                sagemaker("endpoint/*"),
                sagemaker("inference-component/*"),
                sagemaker("model-package/*"),
                sagemaker("model-package-group/*"),
                sagemaker("hub/*"),
                sagemaker("hub-content/*"),
            ],
        },
        {
            "Sid": "ReadSageMaker",
            "Effect": "Allow",
            "Action": ["sagemaker:Describe*", "sagemaker:List*", "sagemaker:Search"],
            "Resource": "*",
        },
        {
            "Sid": "Autoscaling",  # scale-to-zero endpoints
            "Effect": "Allow",
            "Action": [
                "application-autoscaling:RegisterScalableTarget",
                "application-autoscaling:DeregisterScalableTarget",
                "application-autoscaling:PutScalingPolicy",
                "application-autoscaling:Describe*",
                "cloudwatch:PutMetricAlarm",
                "cloudwatch:DeleteAlarms",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:GetMetricData",
                "cloudwatch:DescribeAlarms",
            ],
            "Resource": "*",
        },
        {
            "Sid": "AutoscalingServiceRole",
            "Effect": "Allow",
            "Action": "iam:CreateServiceLinkedRole",
            "Resource": (
                f"{arn}:iam::{acct}:role/aws-service-role/"
                "sagemaker.application-autoscaling.amazonaws.com/*"
            ),
            "Condition": {
                "StringEquals": {
                    "iam:AWSServiceName": "sagemaker.application-autoscaling.amazonaws.com"
                }
            },
        },
        {
            "Sid": "Data",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetBucketLocation",
            ],
            "Resource": [f"{arn}:s3:::skyhammer-{acct}-*", f"{arn}:s3:::skyhammer-{acct}-*/*"],
        },
        {
            "Sid": "Encryption",  # the bootstrap key (alias/skyhammer)
            "Effect": "Allow",
            "Action": ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey", "kms:DescribeKey"],
            "Resource": f"{arn}:kms:*:{acct}:key/*",
            "Condition": {"ForAnyValue:StringEquals": {"kms:ResourceAliases": "alias/skyhammer"}},
        },
        {
            "Sid": "Graders",  # Python and LLM-judge reward functions
            "Effect": "Allow",
            "Action": [
                "lambda:CreateFunction",
                "lambda:GetFunction",
                "lambda:DeleteFunction",
                "lambda:InvokeFunction",
                "lambda:ListTags",
                "lambda:TagResource",
            ],
            "Resource": f"{arn}:lambda:*:{acct}:function:skyhammer-grader-*",
        },
        {
            "Sid": "ListFunctions",
            "Effect": "Allow",
            "Action": "lambda:ListFunctions",
            "Resource": "*",
        },
        {
            "Sid": "PassBootstrapRoles",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": f"{arn}:iam::{acct}:role/{BOOTSTRAP_STACK}-*",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": [
                        "bedrock.amazonaws.com",
                        "sagemaker.amazonaws.com",
                        "lambda.amazonaws.com",
                    ]
                }
            },
        },
        {
            "Sid": "ReadOnlyLookups",  # prices, quotas, training logs, stack outputs
            "Effect": "Allow",
            "Action": [
                "pricing:GetProducts",
                "servicequotas:GetServiceQuota",
                "servicequotas:ListServiceQuotas",
                "logs:DescribeLogStreams",
                "logs:GetLogEvents",
                "cloudformation:DescribeStacks",
                "sts:GetCallerIdentity",
            ],
            "Resource": "*",
        },
    ]
    if bootstrap:
        statements.append(
            {
                "Sid": "Bootstrap",
                "Effect": "Allow",
                "Action": [
                    "cloudformation:CreateStack",
                    "cloudformation:UpdateStack",
                    "cloudformation:DeleteStack",
                ],
                "Resource": f"{arn}:cloudformation:*:{acct}:stack/{BOOTSTRAP_STACK}/*",
            }
        )
    return {"Version": "2012-10-17", "Statement": statements}
