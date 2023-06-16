import React from 'react'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import { useState, useEffect } from 'react'
import Header1 from '../Components/Header1'
import Button from '../Components/Button'
import Header2 from '../Components/Header2'
import FileHandler from '../Components/FileHandler'

function MultipleCustomers() {

    
    let [series, setSeries] = useState([{}])
    const options = {
        series: series,
        title: {
            text: "Churn of Customers"
        },
        chart: {
            type: "line"
        },
        xAxis: {
            categories: []
        }
    }
    const handleChurn = () => {
        let feature_post = []
        let inputFields = document.querySelectorAll("[type=number]");
        for(let i = 0; i < inputFields.length; i++) {
            let name = inputFields[i].id.split(" ")[0]
            let value = inputFields[i].value
            feature_post.push({
                "name": name,
                "value": value
            })
        }
        console.log("feature_post: ", feature_post)

        fetch("http://localhost:8080/predict_dataset", {
            method: "POST",
            mode: "cors",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ feature_post })
        }).then(
            res => res.json()
        ).then(
            data => {
                console.log(data)
                setSeries(data)
            }
        ).catch(
            e => console.error("No prediction yet\n", e)
        )
        
    }

    return (
        <>
            <div className='container'>
                <Header1 className={"text-center lable1 pt-5"}>Predict a dataset of customer's churn!</Header1>
                <FileHandler />
            </div>
            <div className='container'>
                <HighchartsReact highcharts={Highcharts} options={options} />
            </div>
        </>
    )
}

export default MultipleCustomers