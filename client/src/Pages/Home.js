import React from 'react'
import Button from '../components/Button'
import Header1 from '../components/Header1'
import { Link } from 'react-router-dom'
import Heroe from '../components/Heroe'

function Home() {

    return (
        <>
            <div className='' style={{ height: "700px" }}>
                <div className='d-flex flex-column text-center bg-image h-100'>
                    <Heroe
                        display={"display-1"}
                        className={""}
                        children={"Welcome"}
                        subheading={(<>
                                <blockquote className="blockquote">
                                    <p>"Unlocking Customer Retention: Empowering Businesses to Thrive Through Data-driven Churn Analysis."</p>
                                </blockquote>
                                <figcaption className="blockquote-footer">
                                    Founders of <cite title="Source Title">TelCo Generic</cite>
                                </figcaption> </>)}
                        linkButton={"Get Started!"}/>
                </div>
            </div>
        </>
    )
}

export default Home