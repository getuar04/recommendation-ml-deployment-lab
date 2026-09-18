<#
.SYNOPSIS
    Reproducible LOCAL launcher for the Jenkins LAB container that will run Jenkins.lab.

.DESCRIPTION
    Fails closed: refuses to (re)build or start anything unless every safety precondition
    below holds. Never modifies Docker Desktop itself, never modifies the host's own
    kubeconfig (C:\Users\<you>\.kube\config -- only reads it), never creates, deletes or
    resets the jenkins_lab_home volume, and never contacts any Kubernetes context other than
    docker-desktop.

    Steps: preflight checks -> generate a LAB-only kubeconfig (see
    generate_lab_kubeconfig.py) -> build jenkins/Dockerfile.lab -> ensure jenkins_lab_home's
    .kube directory exists -> start the Jenkins LAB container.

    Run from PowerShell:  .\jenkins\run-jenkins-lab.ps1
#>

$ErrorActionPreference = 'Stop'

$JenkinsDir           = $PSScriptRoot
$RepoRoot             = Split-Path -Parent $JenkinsDir
$ImageName            = 'recommendation-ml-jenkins-lab:local'
$ContainerName        = 'jenkins-lab'
$VolumeName           = 'jenkins_lab_home'
$ExpectedKubeContext  = 'docker-desktop'
$HostKubeconfig       = Join-Path $env:USERPROFILE '.kube\config'
$GeneratedDir         = Join-Path $JenkinsDir '.generated'
$GeneratedKubeconfig  = Join-Path $GeneratedDir 'kubeconfig.lab'

function Fail([string]$Message) {
    Write-Host "REFUSING: $Message" -ForegroundColor Red
    exit 1
}

Write-Host '== Jenkins LAB preflight ==' -ForegroundColor Cyan

# 1. Expected kubeconfig exists.
if (-not (Test-Path $HostKubeconfig)) {
    Fail "host kubeconfig not found at $HostKubeconfig."
}

# 2. Docker Desktop context/environment is available.
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Fail 'Docker Desktop daemon is not reachable (`docker info` failed). Is Docker Desktop running?'
}

# 3. Kubernetes context is exactly docker-desktop (checked against the HOST kubeconfig,
#    via the host's own kubectl -- this script never trusts an unverified assumption).
$currentContext = (kubectl config current-context 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($currentContext)) {
    Fail 'could not determine the current kubectl context on the host.'
}
$currentContext = $currentContext.Trim()
if ($currentContext -ne $ExpectedKubeContext) {
    Fail "host kubectl context is '$currentContext', expected exactly '$ExpectedKubeContext'. Refusing to build a lab kubeconfig for anything else."
}

# 4. jenkins_lab_home volume already exists -- this script NEVER creates, deletes, or
#    resets it (existing Jenkins admin user/plugins/jobs must survive).
docker volume inspect $VolumeName *> $null
if ($LASTEXITCODE -ne 0) {
    Fail "Docker volume '$VolumeName' does not exist. This script will not create it -- restore/provision it first, then re-run."
}

# 5. Ports 8080/50000 not already occupied by another container.
foreach ($port in @('8080', '50000')) {
    $inUse = docker ps --format '{{.Names}}||{{.Ports}}' | Select-String -SimpleMatch ":$port->"
    if ($inUse) {
        Fail "port $port is already published by another running container: $inUse"
    }
}

Write-Host 'OK: preflight checks passed.' -ForegroundColor Green

# --- Generate the LAB-only kubeconfig -----------------------------------------------------
# See generate_lab_kubeconfig.py for the full reasoning (loopback address unreachable from
# inside a container; TLS verified via tls-server-name, never --insecure-skip-tls-verify).
# The host's own kubeconfig is only ever READ, never modified. The generator itself refuses
# to run against any context other than docker-desktop and drops every other
# context/cluster/user from the copy it writes, so no non-local (in particular no company)
# Kubernetes target can ever reach the Jenkins container through this file.
Write-Host '== Generating LAB-only kubeconfig ==' -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $GeneratedDir | Out-Null

python "$JenkinsDir\generate_lab_kubeconfig.py" $HostKubeconfig $GeneratedKubeconfig
if ($LASTEXITCODE -ne 0) {
    Fail 'kubeconfig generation failed (see output above).'
}

# Defense in depth: the generated file should never contain a company infrastructure
# reference (it is filtered to only the docker-desktop context/cluster/user, but verify
# anyway rather than assume).
$forbidden = Select-String -Path $GeneratedKubeconfig -Pattern 'g2r|inturium|amazonaws|\.eks\.' -Quiet
if ($forbidden) {
    Fail 'generated kubeconfig unexpectedly contains a company infrastructure reference -- aborting.'
}
Write-Host "OK: $GeneratedKubeconfig generated and scanned clean." -ForegroundColor Green

# --- Build the image -----------------------------------------------------------------------
Write-Host '== Building Jenkins LAB image ==' -ForegroundColor Cyan
docker build -f "$JenkinsDir\Dockerfile.lab" -t $ImageName $RepoRoot
if ($LASTEXITCODE -ne 0) {
    Fail 'docker build failed (see output above).'
}

# --- Ensure jenkins_lab_home\.kube exists, owned by the jenkins user (never touches
#     anything else already in the volume) ---------------------------------------------
docker run --rm -v "${VolumeName}:/var/jenkins_home" $ImageName sh -c 'mkdir -p /var/jenkins_home/.kube'
if ($LASTEXITCODE -ne 0) {
    Fail 'could not prepare /var/jenkins_home/.kube inside the existing volume.'
}

# --- Start the Jenkins LAB container --------------------------------------------------
# - jenkins_lab_home is reused as-is (existing admin user/plugins/jobs survive).
# - /var/run/docker.sock is the HOST Docker Desktop daemon's socket (Docker-outside-of-
#   Docker: this container never runs its own dockerd).
# - the generated kubeconfig is mounted read-only at the exact path kubectl already looks
#   at by default for the jenkins user ($HOME/.kube/config, HOME=/var/jenkins_home).
Write-Host '== Starting Jenkins LAB container ==' -ForegroundColor Cyan
# Remove a previous jenkins-lab container by this same name, if any (the named VOLUME, and
# therefore all Jenkins state, is never touched by this). `docker rm` on a name that does
# not exist writes to stderr and exits non-zero -- expected/harmless here, so swallow it
# explicitly rather than let $ErrorActionPreference='Stop' turn it into a script failure.
try { docker rm -f $ContainerName *> $null } catch {}

docker run -d `
    --name $ContainerName `
    -p 8080:8080 `
    -p 50000:50000 `
    -v "${VolumeName}:/var/jenkins_home" `
    -v /var/run/docker.sock:/var/run/docker.sock `
    -v "${GeneratedKubeconfig}:/var/jenkins_home/.kube/config:ro" `
    $ImageName

if ($LASTEXITCODE -ne 0) {
    Fail 'docker run failed (see output above).'
}

Write-Host 'OK: Jenkins LAB container started. UI: http://localhost:8080' -ForegroundColor Green
Write-Host 'Reminder: this only starts the Jenkins controller. It does not run any pipeline, deploy the app, or touch Kubernetes resources.' -ForegroundColor Yellow
