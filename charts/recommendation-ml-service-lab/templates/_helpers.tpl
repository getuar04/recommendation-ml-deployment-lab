{{/*
Chart base name (nameOverride if set, else the Chart name).
*/}}
{{- define "recommendation-ml-service-lab.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name -- fullnameOverride wins if set (values-lab.yaml always sets it),
otherwise falls back to the standard release-name/chart-name convention.
*/}}
{{- define "recommendation-ml-service-lab.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "recommendation-ml-service-lab.labels" -}}
app.kubernetes.io/name: {{ include "recommendation-ml-service-lab.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels -- must stay immutable across releases (Deployment.spec.selector is
immutable once created), so this deliberately excludes version/managed-by.
*/}}
{{- define "recommendation-ml-service-lab.selectorLabels" -}}
app.kubernetes.io/name: {{ include "recommendation-ml-service-lab.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
