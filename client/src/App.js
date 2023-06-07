import React, { useState, useEffect } from 'react'
import Button from './Components/Button'
import Header1 from './Components/Header1'

function App() {

    const [data, setData] = useState([{}])

    useEffect(() => {
        fetch("/members").then(
            res => res.json()
        ).then(
            data => {
                setData(data)
                console.log(data)
            }
        )
    }, [])

    return (
        <main className='main'>
            <div className='container'>
                <Header1 children={"Customer Churn Analysis"} className="display-1" />
                <input id="pasteData" type="text" placeholder="Paste Excel row here" />
                <Button>Scatter Values</Button>
            </div>
        </main>
    )
}

export default App