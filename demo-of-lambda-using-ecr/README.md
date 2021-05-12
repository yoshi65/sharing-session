# Demo of lambda using ecr

Simple demo. This lambda only response "hello world!".

## How to use

1. Create docker image
    ```sh
    docker build -t func1 .
    ```
1. Run in local
    ```sh
    docker run --rm -p 8080:8080 func1:latest
    ```
1. Test in local
    ```sh
    curl -X POST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{}'
    ```
1. Set env var
    ```sh
    REGION=$(aws configure get region)
    ACCOUNTID=$(aws sts get-caller-identity --output text --query Account)
    ECR_NAME=${your_ecr_name}
    FUNCTION_NAME=${your_function_name}
    ROLE_ARN=${your_role_arn}
    ```
1. Login ECR
    ```sh
    aws ecr get-login-password | docker login --username AWS --password-stdin ${ACCOUNTID}.dkr.ecr.${REGION}.amazonaws.com
    Login Succeeded
    ```
1. Set docker tag
    ```sh
    docker tag func1:latest ${ACCOUNTID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_NAME}:latest
    ```
1. Push docker image
    ```sh
    docker push ${ACCOUNTID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_NAME}:latest
    ```
1. Get Digest
    ```sh
    DIGEST=$(aws ecr list-images --repository-name ${ECR_NAME} --out text --query 'imageIds[?imageTag==`latest`].imageDigest')
    ```
1. Create lambda function
    ```sh
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${ACCOUNTID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_NAME}@${DIGEST} \
        --role ${ROLE_ARN}
    ```
1. Invoke lambda function
    ```sh
    aws lambda invoke --function-name ${FUNCTION_NAME} output ; cat output
    ```

## Ref.
[コンテナ利用者に捧げる AWS Lambda の新しい開発方式 !](https://aws.amazon.com/jp/builders-flash/202103/new-lambda-container-development/?awsf.filter-name=*all)
