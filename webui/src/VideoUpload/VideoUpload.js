import React, { Component, useState } from "react";
import { CreateMultipartUploadCommand,
  UploadPartCommand,
  CompleteMultipartUploadCommand,
  AbortMultipartUploadCommand,
  PutObjectCommand,
} from "@aws-sdk/client-s3";

import Form from "react-bootstrap/Form";
import InputGroup from 'react-bootstrap/InputGroup';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";
import ProgressBar from 'react-bootstrap/ProgressBar';

import './VideoUpload.css';

export class VideoUpload extends Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedVideoFiles: null,
      isUploading: false,
      currentUploadSize: 0,
      totalUploadSize: 0.001,
    };

    this.s3Client = props.s3Client;
    this.bucketName = props.bucketName;
    this.rawFolder = props.rawFolder;
  }

  cleanFileName = (rawFileName) => {
    return rawFileName.replace(/[^a-zA-Z0-9\-\.]/g, '_');
  }

  onFileChange = (event) => {
    this.setState({
      selectedVideoFiles: event.target.files,
    });
  };

  onFilesUpload = async () => {
    this.setState({
      isUploading: true
    });

    var totalSizeToUpload = 0.0
    for (let i = 0; i < this.state.selectedVideoFiles.length; i++) { 
      totalSizeToUpload += this.state.selectedVideoFiles[i].size
    }
    this.setState({
      totalUploadSize: this.state.totalUploadSize + totalSizeToUpload
    });
    
    for (let i = 0; i < this.state.selectedVideoFiles.length; i++) {
      const fileSize = this.state.selectedVideoFiles[i].size
      if(fileSize >=25e6){ // User Multipart upload
        let uploadId;
        try {
          const multipartUpload = await this.s3Client.send(
            new CreateMultipartUploadCommand({
              Bucket: this.bucketName,
              Key: `${this.rawFolder}/${this.cleanFileName(this.state.selectedVideoFiles[i].name)}`,
            }),
          );

          uploadId = multipartUpload.UploadId;

          const uploadPromises = [];
          // Multipart uploads require a minimum size of 5 MB per part.
          const partSize = 10e6
          const numberOfParts = Math.floor(fileSize / partSize)
          const lastPartSize = partSize + fileSize % partSize

          // Upload each part.
          for (let j = 0; j < numberOfParts; j++) {
            const start = j * partSize;
            const end = (j === numberOfParts - 1 ) ? (start + lastPartSize) : (start + partSize)
            uploadPromises.push(
              this.s3Client
                .send(
                  new UploadPartCommand({
                    Bucket: this.bucketName,
                    Key: `${this.rawFolder}/${this.cleanFileName(this.state.selectedVideoFiles[i].name)}`,
                    UploadId: uploadId,
                    Body: this.state.selectedVideoFiles[i].slice(start, end),
                    PartNumber: j + 1,
                  }),
                ).then(singlePartUploadResult => {
                  this.setState({
                    currentUploadSize: this.state.currentUploadSize + ((j === numberOfParts - 1 ) ? lastPartSize : partSize)
                  });
                  return singlePartUploadResult
                })
            );
          }

          const uploadResults = await Promise.all(uploadPromises);

          await this.s3Client.send(
            new CompleteMultipartUploadCommand({
              Bucket: this.bucketName,
              Key: `${this.rawFolder}/${this.cleanFileName(this.state.selectedVideoFiles[i].name)}`,
              UploadId: uploadId,
              MultipartUpload: {
                Parts: uploadResults.map(({ ETag }, i) => ({
                  ETag,
                  PartNumber: i + 1,
                })),
              },
            }),
          );
        } catch (err) {
          console.error(err);

          if (uploadId) {
            const abortCommand = new AbortMultipartUploadCommand({
              Bucket: this.bucketName,
              Key: `${this.rawFolder}/${this.cleanFileName(this.state.selectedVideoFiles[i].name)}`,
              UploadId: uploadId,
            });

            await this.s3Client.send(abortCommand);
          }
        }

      }else{ // Use simple upload
        const command = new PutObjectCommand({
          Bucket: this.bucketName,
          Key: `${this.rawFolder}/${this.cleanFileName(this.state.selectedVideoFiles[i].name)}`,
          Body: this.state.selectedVideoFiles[i],
        });
    
        // Upload file  
        try { 
          await this.s3Client.send(command);
          this.setState({
            currentUploadSize: this.state.currentUploadSize + parseInt(this.state.selectedVideoFiles[i].size)
          });
        } catch (error) {
          console.log(error)
        }
      }
    }

    this.setState({
      isUploading: false,
      selectedVideoFiles: null,
      currentUploadSize: 0,
      totalUploadSize: 0.001,
    });
  };

  render() {
    return (
      <>
        <Row>
          <Col xs={6}>
            <Form.Group controlId="formFile" className="mb-3 no-margin-bottom">
              <InputGroup>
                <InputGroup.Text id="video-upload-text">Upload video files</InputGroup.Text>
                <Form.Control
                  type="file"
                  accept="video/mp4"
                  onChange={this.onFileChange}
                  multiple={true}
                />
                <Button
                  onClick={this.onFilesUpload.bind(this)}
                  variant={!this.state.isUploading ? 'success' : 'primary'}
                  id="upload-video"
                  disabled={!this.state.selectedVideoFiles || this.state.isUploading}
                >
                  {this.state.isUploading ? 'Uploading . . .' : 'Upload'}
                </Button>
              </InputGroup>
            </Form.Group>
          </Col>
          <Col xs={6}>
            <p className="no-margin-bottom" align="left">Alternatively, upload to Amazon S3 bucket <i>"{this.bucketName}"</i> inside folder <i>"{this.rawFolder}"</i>.</p>
          </Col>
        </Row>
        { this.state.isUploading ? <Row><Col> <ProgressBar animated now={Math.floor(this.state.currentUploadSize / parseInt(this.state.totalUploadSize) * 100)} label={`${Math.floor(this.state.currentUploadSize / parseInt(this.state.totalUploadSize) * 100)}%`} /></Col></Row> : "" }
        { this.state.isUploading && Math.floor(this.state.currentUploadSize / parseInt(this.state.totalUploadSize) * 100) > 50 ? <Row><Col><p className="no-margin-bottom" align="left">When upload finishes, you can reload page or re-search the videos to refresh the video list.</p></Col></Row> : "" }
      </>
    );
  }
}
